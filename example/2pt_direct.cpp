#include "fpsolver.h"

using pointsVec = std::vector<cv::Point2f>;
using matVec = std::vector<cv::Mat>;

int main() {
  // needs to be exactly two correspondences
  pointsVec points1_est = {cv::Point2f(1652, 469), cv::Point2f(1642, 656)};
  pointsVec points2_est = {cv::Point2f(1712, 453), cv::Point2f(1739, 696)};

  cv::Mat k = (cv::Mat_<double>(3, 3) << 1280.7, 0.0, 969.4257, 0.0, 1281.2,
               639.7227, 0.0, 0.0, 1.0); // intrinsic matrix example

  matVec E_cand = PM::findEssential(points1_est, points2_est, k);

  for (auto &E : E_cand)
    std::cout << E << "\n";
}
