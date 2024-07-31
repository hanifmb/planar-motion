#include "fpsolver.h"
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

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

int main() {
  std::vector<cv::Point2f> points1;
  std::vector<cv::Point2f> points2;

  points1.push_back(cv::Point2f(1652, 469));
  points1.push_back(cv::Point2f(1642, 656));
  points2.push_back(cv::Point2f(1712, 453));
  points2.push_back(cv::Point2f(1739, 696));

  // intrinsic matrix for both camera

  cv::Mat k = (cv::Mat_<double>(3, 3) << 1280.7, 0.0, 969.4257, 0.0, 1281.2,
               639.7227, 0.0, 0.0, 1.0);

  std::vector<cv::Mat> essentialMatrices =
      PM::findEssential(points1, points2, k);

  cv::Mat E = essentialMatrices[2];
  cv::Mat F = k.t().inv() * E * k.inv();

  std::string imagepath1 =
      "/home/batman/fun/pose-from-planar-motion/input_images/1.jpg";
  std::string imagepath2 =
      "/home/batman/fun/pose-from-planar-motion/input_images/2.jpg";
  cv::Mat img1 = cv::imread(imagepath1);
  cv::Mat img2 = cv::imread(imagepath2);

  points1.push_back(cv::Point2f(1122, 361));
  points1.push_back(cv::Point2f(1086, 408));
  points2.push_back(cv::Point2f(997, 270));
  points2.push_back(cv::Point2f(946, 342));

  for (auto &e : essentialMatrices)
    std::cout << "E: \n" << e << "\n\n";

  // drawEpipolarLines(img1, img2, F_cv, points1, points2);
  drawEpipolarLines(img1, img2, F, points1, points2);
  return 0;
}
