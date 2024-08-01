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

void drawEpilines(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &F,
                  const pointsVec &points1, const pointsVec &points2) {

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

std::tuple<pointsVec, pointsVec> matchFeatures(const cv::Mat &img1,
                                               const cv::Mat &img2,
                                               float ratio_thresh,
                                               int num_top_matches) {
  // Initialize the ORB detector
  cv::Ptr<cv::ORB> orb = cv::ORB::create();

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

  // Feature matching
  float ratio_thresh = 0.85;
  int num_top_matches = 100;
  pointsVec points1, points2;
  std::tie(points1, points2) =
      matchFeatures(img1, img2, ratio_thresh, num_top_matches);

  // Generate two random integers
  int num1, num2;
  std::tie(num1, num2) = randomizeInts(0, points1.size() - 1);

  // Push back two random visual corespondences
  pointsVec points1_est = {points1[num1], points1[num2]};
  pointsVec points2_est = {points2[num1], points2[num2]};

  cv::Mat k = (cv::Mat_<double>(3, 3) << 1280.7, 0.0, 969.4257, 0.0, 1281.2,
               639.7227, 0.0, 0.0, 1.0);

  // Find fundamental matrix with ransac
  int max_iter = 1000;
  double threshold = 350;
  double confidence = 99.0;
  PM::RANSACFundam ransac(max_iter, threshold, confidence);

  cv::Mat F = PM::findFundam(points1, points2, k, ransac);

  // Visualize the epipolar lines
  drawEpilines(img1, img2, F, points1, points2);

  return 0;
}
