#include "fpsolver.h"
#include <array>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>

void drawEpipolarLines(const cv::Mat &img1, const cv::Mat &img2,
                       const cv::Mat &F,
                       const std::vector<cv::Point2f> &points1,
                       const std::vector<cv::Point2f> &points2) {

  cv::Mat img1_copy, img2_copy;
  img1.copyTo(img1_copy);
  img2.copyTo(img2_copy);

  // Compute the epilines for points in the first image
  std::vector<cv::Vec3f> lines1, lines2;
  cv::computeCorrespondEpilines(points1, 1, F,
                                lines1); // Compute epilines in the second image
                                         // for points from the first image
  cv::computeCorrespondEpilines(points2, 2, F,
                                lines2); // Compute epilines in the first image
                                         // for points from the second image

  // Draw the epipolar lines
  cv::RNG rng;
  for (size_t i = 0; i < points1.size(); i++) {
    cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                                  rng.uniform(0, 255));

    // Epipolar lines in the first image (corresponding to points in the second
    // image)
    cv::line(img1_copy, cv::Point(0, -lines2[i][2] / lines2[i][1]),
             cv::Point(img2_copy.cols,
                       -(lines2[i][2] + lines2[i][0] * img2_copy.cols) /
                           lines2[i][1]),
             color);
    cv::circle(img1_copy, points1[i], 5, color, -1);

    // Epipolar lines in the second image (corresponding to points in the first
    // image)
    cv::line(img2_copy, cv::Point(0, -lines1[i][2] / lines1[i][1]),
             cv::Point(img1_copy.cols,
                       -(lines1[i][2] + lines1[i][0] * img1_copy.cols) /
                           lines1[i][1]),
             color);
    cv::circle(img2_copy, points2[i], 5, color, -1);
  }

  // Display the images with epipolar lines
  cv::imshow("Image 1 with Epipolar Lines", img1_copy);
  cv::imshow("Image 2 with Epipolar Lines", img2_copy);
  cv::waitKey(0);
}

std::string _to2Sig(float number) {
  // reduce float to 2 significant digits, return as a string
  if (number == 0.0)
    return "0.0"; // Handle zero separately

  std::stringstream ss;
  ss << std::scientific << std::setprecision(2) << number;
  return ss.str();
}

int main() {
  std::string imagepath1 =
      "/home/batman/fun/pose-from-planar-motion/input_images/3.jpg";
  std::string imagepath2 =
      "/home/batman/fun/pose-from-planar-motion/input_images/4.jpg";
  // Load the images
  cv::Mat img1 = cv::imread(imagepath1, cv::IMREAD_COLOR);
  cv::Mat img2 = cv::imread(imagepath2, cv::IMREAD_COLOR);
  if (img1.empty() || img2.empty()) {
    std::cerr << "Could not open or find the images" << std::endl;
    return -1;
  }

  cv::Mat img1Gray, img2Gray;
  cv::cvtColor(img1, img1Gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(img2, img2Gray, cv::COLOR_BGR2GRAY);

  // Initialize the ORB detector
  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  // Detect keypoints and compute descriptors
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;
  orb->detectAndCompute(img1Gray, cv::noArray(), keypoints1, descriptors1);
  orb->detectAndCompute(img2Gray, cv::noArray(), keypoints2, descriptors2);

  // Use BFMatcher to match descriptors
  cv::BFMatcher bfMatcher(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch>> knnMatches;
  bfMatcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

  // Apply ratio test
  const float ratio_thresh = 0.85;
  std::vector<cv::DMatch> goodMatches;
  for (size_t i = 0; i < knnMatches.size(); i++) {
    if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
      goodMatches.push_back(knnMatches[i][0]);
    }
  }

  // Sort matches by distance
  std::sort(goodMatches.begin(), goodMatches.end());

  // Limit the number of matches to the top N matches
  const int numTopMatches = 50;
  if (goodMatches.size() > numTopMatches) {
    goodMatches.resize(numTopMatches);
  }

  // Accumulating matched features into points1 and points2
  std::vector<cv::Point2f> points1, points2;
  int matchesSize = goodMatches.size();
  points1.reserve(matchesSize), points2.reserve(matchesSize);
  for (size_t i = 0; i < goodMatches.size(); i++) {
    cv::Point2f pt1 = keypoints1[goodMatches[i].queryIdx].pt;
    cv::Point2f pt2 = keypoints2[goodMatches[i].trainIdx].pt;
    points1.push_back(pt1);
    points2.push_back(pt2);
  }

  // Generate two random integers
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis2(0, points1.size() - 1);

  std::set<int> numbers;
  while (numbers.size() < 2) {
    int rand_num = dis2(gen);
    numbers.insert(rand_num);
  }

  auto it = numbers.begin();
  int num1, num2;
  num1 = *it;
  ++it;
  num2 = *it;

  // Push back two random visual corespondences
  std::vector<cv::Point2f> points1_est = {points1[num1], points1[num2]};
  std::vector<cv::Point2f> points2_est = {points2[num1], points2[num2]};

  // Find the essential matrix
  cv::Mat k = (cv::Mat_<double>(3, 3) << 1280.7, 0.0, 969.4257, 0.0, 1281.2,
               639.7227, 0.0, 0.0, 1.0);
  std::vector<cv::Mat> E = PM::findEssential(points1_est, points2_est, k);

  /*
  cv::Mat F = k.t().inv() * E[0] * k.inv();
  cv::Mat F_normalized = PM::_normalizeF(F);

  drawEpipolarLines(img1, img2, F, points1, points2);

  cv::Mat err(points1.size(), 1, CV_64FC1);
  PM::_computeError(points1, points2, F_normalized, err);
  int inlierCount = PM::_countBelow(err, 170);
  std::cout << err << "\n";
  std::cout << inlierCount << "\n";
  */

  //

  int max_iter = 3000;
  double threshold = 150;
  double confidence = 99.0;
  PM::RANSACFundam ransac(max_iter, threshold, confidence);

  cv::Mat F = PM::findEssential(points1, points2, k, ransac);
  cv::Mat F_normalized = PM::_normalizeF(F);
  drawEpipolarLines(img1, img2, F_normalized, points1, points2);

  return 0;
}
