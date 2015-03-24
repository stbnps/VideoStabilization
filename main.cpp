//	Copyright (c) 2015, Esteban Pardo Sánchez
//	All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without modification,
//	are permitted provided that the following conditions are met:
//
//	1. Redistributions of source code must retain the above copyright notice, this
//	list of conditions and the following disclaimer.
//
//	2. Redistributions in binary form must reproduce the above copyright notice,
//	this list of conditions and the following disclaimer in the documentation and/or
//	other materials provided with the distribution.
//
//	3. Neither the name of the copyright holder nor the names of its contributors
//	may be used to endorse or promote products derived from this software without
//	specific prior written permission.
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
//	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//	LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
//	ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d.hpp"

const std::string fileName = "PoliceFootage.mov"; // Name of your input file

double resizeFactor = 0.25; // Resize the input frames by this factor

double descriptorDistanceThreshold = 7.0; // Good matches are below "descriptorDistanceThreshold" times the minimum distance

const double correctionAmount = 0.8; // Set it between 0 and 1: The higher the smoother the footage will be

const float akazeThreshold = 4e-4; // The higher the less features it will track

int main() {

	std::vector<cv::KeyPoint> kptsCurrent, kptsPrevious;
	cv::Mat descCurrent, descPrevious;

	// AKAZE features:
	// Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces.
	// Pablo F. Alcantarilla, Jesús Nuevo and Adrien Bartoli.
	// In British Machine Vision Conference (BMVC), Bristol, UK, September 2013.
	cv::Ptr<cv::FeatureDetector> akaze = cv::AKAZE::create(
			cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, akazeThreshold, 4, 4,
			cv::KAZE::DIFF_PM_G2);

	// Descriptor brute force matching using hamming distance
	cv::BFMatcher matcher(cv::NORM_HAMMING);

	std::vector<std::vector<cv::DMatch> > nn_matches;

	std::vector<cv::DMatch> matches;

	std::vector<cv::DMatch> goodMatches;

	cv::VideoCapture cap(fileName);

	if (!cap.isOpened()) {
		return -1;
	}

	cv::Mat frame;
	cv::Mat previousFrame;
	cap >> frame;

	cv::resize(frame, frame,
			cv::Size(frame.cols * resizeFactor, frame.rows * resizeFactor));

	akaze->detectAndCompute(frame, cv::noArray(), kptsCurrent, descCurrent);

	// Kalman filter to smooth the homographies over time
	cv::KalmanFilter kf(8, 8, 0, CV_64F);

	cv::setIdentity(kf.measurementMatrix);

	cv::namedWindow("Input");
	cv::namedWindow("Stabilized");
	cv::moveWindow("Stabilized", 400, 0);

	cv::Mat frameCorrected = frame.clone();

	// The initial homography is the identity transformation
	cv::Mat previousH(3, 3, CV_64F);
	cv::setIdentity(previousH);

	bool kfInitialized = false;
	for (;;) {

		previousFrame = frame.clone();
		kptsPrevious = kptsCurrent;

		descPrevious = descCurrent.clone();

		cap >> frame;
		cv::resize(frame, frame,
				cv::Size(frame.cols * resizeFactor, frame.rows * resizeFactor));

		kptsCurrent.clear();
		descCurrent = cv::Mat();

		std::vector<cv::Point2f> ptsCurrent;
		std::vector<cv::Point2f> ptsPrevious;

		akaze->detectAndCompute(frame, cv::noArray(), kptsCurrent, descCurrent);

		// We havent retrieved enough keypoints
		if (kptsCurrent.size() < 20) {
			continue;
		}

		matcher.match(descCurrent, descPrevious, matches);

		double minDistance = 100;
		for (unsigned int i = 0; i < matches.size(); i++) {
			if (matches[i].distance < minDistance) {
				minDistance = matches[i].distance;
			}

		}

		for (unsigned int i = 0; i < matches.size(); i++) {
			if (matches[i].distance
					< std::max(25.0,
							descriptorDistanceThreshold * minDistance)) {
				ptsCurrent.push_back(kptsCurrent[matches[i].queryIdx].pt);
				ptsPrevious.push_back(kptsPrevious[matches[i].trainIdx].pt);
			}
		}

		// Wee need at least 4 points on each image to calculate an homography
		// Set the threshold higher because of RANSAC
		if (ptsCurrent.size() < 10 || ptsPrevious.size() < 10) {
			continue;
		}
		cv::Mat H = cv::findHomography(ptsPrevious, ptsCurrent, cv::RANSAC);

		// Initialize the Kalman filter state
		// Only performed the first time an homography is calculated
		if (!kfInitialized) {
			kf.statePre = H.reshape(1, 9).rowRange(0, 8).clone();
			kf.statePost = H.reshape(1, 9).rowRange(0, 8).clone();
			kfInitialized = true;
		}

		// The homography calculation has failed
		if (H.cols < 3) {
			continue;
		}

		// Kalman filter loop: predict and correct
		cv::Mat predicted = kf.predict();
		cv::Mat HEstimated = kf.correct(H.reshape(1, 9).rowRange(0, 8));
		HEstimated.push_back(cv::Mat(1, 1, CV_64F, cv::Scalar(1)));

		HEstimated = HEstimated.reshape(1, 3);

		// This controls the ammount of smoothing
		// The more weight previousH has, the more stable the footage will be
		cv::addWeighted(previousH, correctionAmount, cv::Mat::eye(3, 3, CV_64F),
				1 - correctionAmount, 0, previousH);

		cv::Mat HEstimated_ptsPrevious; // H_smoothed * previous_points
		cv::Mat HPrevious_ptsPrevious; //  H_used_to correct_previous_frame * previous_points

		// The homography between:
		// The previous points transformed with the smoothed homography and
		// The prevous points transformed with the previous correction homography
		// Is the homography that corrects the current frame

		cv::perspectiveTransform(ptsPrevious, HEstimated_ptsPrevious,
				HEstimated);
		cv::perspectiveTransform(ptsPrevious, HPrevious_ptsPrevious, previousH);

		cv::Mat HCorrection = cv::findHomography(HEstimated_ptsPrevious,
				HPrevious_ptsPrevious, cv::RANSAC);

		cv::warpPerspective(frame, frameCorrected, HCorrection,
				cv::Size(frame.cols, frame.rows), cv::INTER_CUBIC);

		previousH = HCorrection.clone();

		cv::Mat frameVisualization;
		cv::Mat frameCorrectedVisualization;

		cv::resize(frameCorrected, frameCorrectedVisualization,
				cv::Size(frameCorrected.cols / (resizeFactor),
						frameCorrected.rows / (resizeFactor)));
		cv::resize(frame, frameVisualization,
				cv::Size(frame.cols / (resizeFactor),
						frame.rows / (resizeFactor)));

		imshow("Stabilized", frameCorrectedVisualization);

		imshow("Input", frameVisualization);

		int keyCode = cv::waitKey(60);
		if (keyCode > 0) {
			return 0;
		}

	}

}
