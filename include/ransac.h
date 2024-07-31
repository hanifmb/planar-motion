/*
#ifndef RANSAC_H
#define RANSAC_H

#include "fpsolver.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

class RANSACFundam {
public:
  RANSACFundam(int maxIterations, double threshold, double confidence)
      : maxIterations(maxIterations), threshold(threshold),
        confidence(confidence) {
    std::srand(std::time(0));
  }

  // Fit a line to the given points using RANSAC
  cv::Mat run(const std::vector<cv::Point2f> &points1,
              const std::vector<cv::Point2f> &points2, const cv::Mat k) {
    int bestInlierCount = 0;
    cv::Mat bestFundam;
    int totalPoints = points1.size();

    for (int i = 0; i < maxIterations; ++i) {
      // Randomly select two points to calculate fundamental matrices
      int idx1 = std::rand() % totalPoints;
      int idx2 = std::rand() % totalPoints;

      if (idx1 == idx2) {
        --i;
        continue; // Avoid selecting the same point twice
      }

      std::vector<cv::Point2f> points1_est = {points1[idx1], points1[idx2]};
      std::vector<cv::Point2f> points2_est = {points2[idx1], points2[idx2]};

      // Calculate line parameters (y = mx + b)
      std::vector<cv::Mat> E = PM::findEssential(points1_est, points2_est, k);

      // Evaluate multiple candidates of essential matrices
      for (int i = 0; i < E.size(); ++i) {
        cv::Mat F = k.t().inv() * E[i] * k.inv();
        cv::Mat err(points1.size(), 1, CV_64FC1);
        PM::_computeError(points1, points2, F, err);

        // todo: calculate inlierCount
        int inlierCount = _countBelow(err, threshold);
        if (inlierCount > bestInlierCount) {
          bestInlierCount = inlierCount;
          bestFundam = F;
        }

        // Check if the model is good enough
        double inlierRatio = static_cast<double>(bestInlierCount) / totalPoints;
        if (inlierRatio > confidence)
          return F;
      }
    }

    return bestFundam;
  }

  int _countBelow(const cv::Mat &err, double threshold) {
    cv::Mat mask;
    cv::threshold(err, mask, threshold, 255, cv::THRESH_BINARY_INV);
    int count = cv::countNonZero(mask);
    return count;
  }

private:
  int maxIterations;
  double distanceThreshold;
  double confidence;
  double threshold;
};

#endif // RANSAC_H
*/
