#include "fpsolver.h"
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

// Function to convert Eigen::Vector2d to cv::Point2f
cv::Point2f eigenVectorToPoint2f(const Eigen::Vector2d &eigenVec) {
  return cv::Point2f(static_cast<float>(eigenVec.x()),
                     static_cast<float>(eigenVec.y()));
}

// Function to convert std::vector<Eigen::Vector2d> to std::vector<cv::Point2f>
std::vector<cv::Point2f>
eigenVectorsToPoints(const std::vector<Eigen::Vector2d> &eigenVecs) {
  std::vector<cv::Point2f> points;
  for (const auto &vec : eigenVecs) {
    points.push_back(eigenVectorToPoint2f(vec));
  }
  return points;
}

void drawEpipolarLines(const cv::Mat &img1, const cv::Mat &img2,
                       const cv::Mat &F,
                       const std::vector<Eigen::Vector2d> &eigenPoints1,
                       const std::vector<Eigen::Vector2d> &eigenPoints2) {
  // Convert Eigen::Vector2d to cv::Point2f
  std::vector<cv::Point2f> points1 = eigenVectorsToPoints(eigenPoints1);
  std::vector<cv::Point2f> points2 = eigenVectorsToPoints(eigenPoints2);

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
  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  Eigen::Matrix3d k;

  points1.push_back(Eigen::Vector2d(1652, 469));
  points1.push_back(Eigen::Vector2d(1642, 656));
  points2.push_back(Eigen::Vector2d(1712, 453));
  points2.push_back(Eigen::Vector2d(1739, 696));

  // intrinsic matrix for both camera
  k << 1280.7, 0.0, 969.4257, 0.0, 1281.2, 639.7227, 0.0, 0.0, 1.0;

  std::vector<Eigen::MatrixXd> essentialMatrices =
      PM::findEssential(points1, points2, k);

  Eigen::MatrixXd E = essentialMatrices[2];
  Eigen::MatrixXd F = k.transpose().inverse() * E * k.inverse();

  cv::Mat F_cv;
  cv::eigen2cv(F, F_cv);

  std::string imagepath1 =
      "/home/batman/fun/pose-from-planar-motion/input_images/1.jpg";
  std::string imagepath2 =
      "/home/batman/fun/pose-from-planar-motion/input_images/2.jpg";
  cv::Mat img1 = cv::imread(imagepath1);
  cv::Mat img2 = cv::imread(imagepath2);

  points1.push_back(Eigen::Vector2d(1122, 361));
  points1.push_back(Eigen::Vector2d(1086, 408));
  points2.push_back(Eigen::Vector2d(997, 270));
  points2.push_back(Eigen::Vector2d(946, 342));

  for (auto &E : essentialMatrices) {
    cv::Mat R1, R2, t, E_cv;
    cv::eigen2cv(E, E_cv);
    cv::decomposeEssentialMat(E_cv, R1, R2, t);
    cv::Mat min_t = -t;

    // camera 1 is the reference
    cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F); // [I | 0]

    // four possible fundamental matrices for the second camera
    cv::Mat P2_1 = cv::Mat::zeros(3, 4, CV_64F);
    R1.copyTo(P2_1(cv::Rect(0, 0, 3, 3)));
    t.copyTo(P2_1(cv::Rect(3, 0, 1, 3)));

    cv::Mat P2_2 = cv::Mat::zeros(3, 4, CV_64F);
    R1.copyTo(P2_2(cv::Rect(0, 0, 3, 3)));
    min_t.copyTo(P2_2(cv::Rect(3, 0, 1, 3)));

    cv::Mat P2_3 = cv::Mat::zeros(3, 4, CV_64F);
    R2.copyTo(P2_3(cv::Rect(0, 0, 3, 3)));
    t.copyTo(P2_3(cv::Rect(3, 0, 1, 3)));

    cv::Mat P2_4 = cv::Mat::zeros(3, 4, CV_64F);
    R2.copyTo(P2_4(cv::Rect(0, 0, 3, 3)));
    min_t.copyTo(P2_4(cv::Rect(3, 0, 1, 3)));

    std::vector<cv::Point2f> cvpoints1 = eigenVectorsToPoints(points1);
    std::vector<cv::Point2f> cvpoints2 = eigenVectorsToPoints(points2);

    cv::Mat points4D;
    cv::triangulatePoints(P1, P2_4, cvpoints1, cvpoints2, points4D);

    /*
    Eigen::Vector3d x(1642, 656, 1);
    Eigen::Vector3d x_prime(1739, 696, 1);
    Eigen::MatrixXd res = x_prime.transpose() * F * x;
    std::cout << res << "\n";
    */
  }

  for (auto &e : essentialMatrices)
    std::cout << "E: \n" << e << "\n\n";

  // drawEpipolarLines(img1, img2, F_cv, points1, points2);
  drawEpipolarLines(img1, img2, F_cv, points1, points2);
  return 0;
}
