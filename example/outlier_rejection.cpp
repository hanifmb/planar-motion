#include "fpsolver.h"
#include <array>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <tuple>

using pointsVec = std::vector<cv::Point2f>;

std::tuple<int, int> randomizeInts(int min_int, int max_int) {
  // Generate two random integers
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis2(min_int, max_int);

  std::set<int> numbers;
  while (numbers.size() < 2) {
    int rand_num = dis2(gen);
    numbers.insert(rand_num);
  }

  auto it = numbers.begin();
  int num1 = *it;
  ++it;
  int num2 = *it;

  return std::make_tuple(num1, num2);
}

std::tuple<pointsVec, pointsVec> matchFeatures(const cv::Mat &img1,
                                               const cv::Mat &img2,
                                               float ratio_thresh,
                                               int num_top_matches) {

  // Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization)
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));

  // Convert images to grayscale if they are not already
  cv::Mat img1Gray, img2Gray;
  if (img1.channels() > 1) {
    cv::cvtColor(img1, img1Gray, cv::COLOR_BGR2GRAY);
  } else {
    img1Gray = img1;
  }
  if (img2.channels() > 1) {
    cv::cvtColor(img2, img2Gray, cv::COLOR_BGR2GRAY);
  } else {
    img2Gray = img2;
  }

  // Apply CLAHE to the grayscale images
  cv::Mat img1Equalized, img2Equalized;
  clahe->apply(img1Gray, img1Equalized);
  clahe->apply(img2Gray, img2Equalized);

  // Initialize the SIFT detector with custom parameters
  cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

  // Detect keypoints and compute descriptors
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;
  sift->detectAndCompute(img1Equalized, cv::noArray(), keypoints1,
                         descriptors1);
  sift->detectAndCompute(img2Equalized, cv::noArray(), keypoints2,
                         descriptors2);

  // Use BFMatcher to match descriptors
  cv::BFMatcher bfMatcher(cv::NORM_L2);
  std::vector<std::vector<cv::DMatch>> knnMatches;
  bfMatcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

  // Apply ratio test
  std::vector<cv::DMatch> goodMatches;
  for (const auto &match : knnMatches) {
    if (match[0].distance < ratio_thresh * match[1].distance) {
      goodMatches.push_back(match[0]);
    }
  }

  // Sort matches by distance
  std::sort(goodMatches.begin(), goodMatches.end());

  // Limit the number of matches to the top N matches
  if (goodMatches.size() > num_top_matches) {
    goodMatches.resize(num_top_matches);
  }

  // Accumulating matched features into points1 and points2
  pointsVec points1, points2;
  int matchesSize = goodMatches.size();
  points1.reserve(matchesSize);
  points2.reserve(matchesSize);
  for (const auto &match : goodMatches) {
    cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
    cv::Point2f pt2 = keypoints2[match.trainIdx].pt;
    points1.push_back(pt1);
    points2.push_back(pt2);
  }

  return std::make_tuple(points1, points2);
}

void drawMatches(const cv::Mat &image1, const cv::Mat &image2,
                 const std::vector<cv::Point2f> &points1,
                 const std::vector<cv::Point2f> &points2, const cv::Mat &err,
                 double threshold) {
  // Check if images and points vectors are valid
  if (image1.empty() || image2.empty()) {
    std::cerr << "Error: One or both images are empty!" << std::endl;
    return;
  }
  if (points1.size() != points2.size()) {
    std::cerr << "Error: Mismatch in the number of feature points!"
              << std::endl;
    return;
  }

  // Create a combined image with images stacked vertically
  cv::Mat combinedImage(image1.rows + image2.rows,
                        std::max(image1.cols, image2.cols), image1.type());
  image1.copyTo(combinedImage(cv::Rect(0, 0, image1.cols, image1.rows)));
  image2.copyTo(
      combinedImage(cv::Rect(0, image1.rows, image2.cols, image2.rows)));

  // Draw lines for feature matches
  const double *errPtr = err.ptr<double>();

  for (size_t i = 0; i < points1.size(); ++i) {
    cv::Point2f p1 = points1[i];
    cv::Point2f p2 = points2[i];

    cv::Scalar color;
    if (errPtr[i] < threshold * threshold) {
      color = cv::Scalar(0, 255, 0);
    } else {
      color = cv::Scalar(0, 0, 255);
    }

    // Adjust coordinates to the new combined image layout
    cv::Point2f p1_in_combined(p1.x, p1.y);
    cv::Point2f p2_in_combined(p2.x, p2.y + image1.rows);

    // Draw line
    cv::line(combinedImage, p1_in_combined, p2_in_combined, color, 1);

    // Draw circles around points
    cv::circle(combinedImage, p1_in_combined, 2, color, -1);
    cv::circle(combinedImage, p2_in_combined, 2, color, -1);
  }

  // Display the combined image
  cv::imshow("Feature Matches", combinedImage);

  cv::imwrite("../../results/tmp/inlier_output.png", combinedImage);
  cv::waitKey(0); // Wait for a key press to close the window
}

int main() {
  std::string imagepath1 = "../../dataset/0000000010.png";
  std::string imagepath2 = "../../dataset/0000000015.png";
  // Load the images
  cv::Mat img1 = cv::imread(imagepath1, cv::IMREAD_COLOR);
  cv::Mat img2 = cv::imread(imagepath2, cv::IMREAD_COLOR);
  if (img1.empty() || img2.empty()) {
    std::cerr << "Could not open or find the images" << std::endl;
    return -1;
  }

  // Feature matching
  float ratio_thresh = 0.95;
  int num_top_matches = 350;

  pointsVec points1, points2;
  std::tie(points1, points2) =
      matchFeatures(img1, img2, ratio_thresh, num_top_matches);

  cv::Mat k = (cv::Mat_<double>(3, 3) << 984.2439, 0.0, 690.0000, 0.0, 980.8141,
               233.1966, 0.0, 0.0, 1.0);

  // Find fundamental matrix with ransac
  int max_iter = 10000;
  double threshold = 20;
  double confidence = 99.0;
  PM::RANSACFundam ransac(max_iter, threshold, confidence);

  cv::Mat F = PM::findFundam(points1, points2, k, ransac);

  size_t size = points1.size();
  cv::Mat err(size, 1, CV_64FC1);
  PM::computeError(points1, points2, F, err);

  // Draw matches
  drawMatches(img1, img2, points1, points2, err, threshold);

  return 0;
}
