#ifndef FPSOLVER_H
#define FPSOLVER_H

#include <array>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace PM {

using pointVec = std::vector<cv::Point2f>;
using matVec = std::vector<cv::Mat>;

matVec findEssential(const pointVec &original_pixels,
                     const pointVec &corresponding_pixels, const cv::Mat &k);
void computeError(pointVec m1, pointVec m2, cv::Mat model, cv::Mat &err);
class RANSACFundam {

public:
  RANSACFundam(int maxIterations, double threshold, double confidence)
      : maxIterations(maxIterations), threshold(threshold * threshold),
        confidence(confidence) {
    std::srand(std::time(0));
  }

  cv::Mat run(const pointVec &points1, const pointVec &points2,
              const cv::Mat k) const {
    int bestInlierCount = 0;
    cv::Mat bestFundam;
    int totalPoints = points1.size();
    cv::Mat besterr;

    for (int i = 0; i < maxIterations; ++i) {
      // Randomly select two points to calculate fundamental matrices
      int idx1 = std::rand() % totalPoints;
      int idx2 = std::rand() % totalPoints;

      if (idx1 == idx2) {
        --i;
        continue; // Avoid selecting the same point twice
      }

      pointVec points1_est = {points1[idx1], points1[idx2]};
      pointVec points2_est = {points2[idx1], points2[idx2]};

      matVec E = findEssential(points1_est, points2_est, k);

      // Evaluate multiple candidates of essential matrices
      for (size_t i = 0; i < E.size(); ++i) {
        cv::Mat F = k.t().inv() * E[i] * k.inv();
        cv::Mat err(points1.size(), 1, CV_64FC1);
        computeError(points1, points2, F, err);

        // Count the number of inliers
        int inlierCount = countBelow(err, threshold);
        if (inlierCount > bestInlierCount) {
          besterr = err;
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

private:
  int maxIterations;
  double confidence;
  double threshold;

  int countBelow(const cv::Mat &err, double threshold) const {
    cv::Mat mask;
    cv::threshold(err, mask, threshold, 255, cv::THRESH_BINARY_INV);
    int count = cv::countNonZero(mask);
    return count;
  }
};

cv::Mat _backProjectPx(const cv::Vec3d &px, const cv::Mat &intrinsic_matrix) {
  // Returning ray vector in camera coordinate from an image pixel
  return intrinsic_matrix.inv() * px;
}

cv::Mat _normalizeF(const cv::Mat &F) {
  double norm_F = cv::norm(F, cv::NORM_L2);
  cv::Mat F_normalized = F / norm_F;
  return F_normalized;
}

void computeError(pointVec m1, pointVec m2, cv::Mat model, cv::Mat &err) {

  const double *F = model.ptr<double>();
  double *errPtr = err.ptr<double>();

  int count = m1.size();
  for (int i = 0; i < count; i++) {
    double a, b, c, d1, d2, s1, s2;

    a = F[0] * m1[i].x + F[1] * m1[i].y + F[2];
    b = F[3] * m1[i].x + F[4] * m1[i].y + F[5];
    c = F[6] * m1[i].x + F[7] * m1[i].y + F[8];

    s2 = 1. / (a * a + b * b);
    d2 = m2[i].x * a + m2[i].y * b + c;

    a = F[0] * m2[i].x + F[3] * m2[i].y + F[6];
    b = F[1] * m2[i].x + F[4] * m2[i].y + F[7];
    c = F[2] * m2[i].x + F[5] * m2[i].y + F[8];

    s1 = 1. / (a * a + b * b);
    d1 = m1[i].x * a + m1[i].y * b + c;

    errPtr[i] = std::max(d1 * d1 * s1, d2 * d2 * s2);
  }
}

matVec findEssential(const pointVec &original_pixels,
                     const pointVec &corresponding_pixels, const cv::Mat &k) {

  // Normalize the point correspondences
  std::vector<cv::Vec3d> op_cam, cp_cam;
  size_t pixels_size = original_pixels.size();
  op_cam.reserve(pixels_size);
  cp_cam.reserve(pixels_size);

  for (int i = 0; i < 2; ++i) {
    // Convert to homogeneous points
    cv::Vec3d op_h(original_pixels[i].x, original_pixels[i].y, 1.0);
    cv::Vec3d cp_h(corresponding_pixels[i].x, corresponding_pixels[i].y, 1.0);

    // Backproject the homogeneous points to camera points
    op_cam.push_back(_backProjectPx(op_h, k));
    cp_cam.push_back(_backProjectPx(cp_h, k));
  }

  double x1_l = op_cam[0][0], y1_l = op_cam[0][1], z1_l = op_cam[0][2];
  double x2_l = op_cam[1][0], y2_l = op_cam[1][1], z2_l = op_cam[1][2];
  double x1_r = cp_cam[0][0], y1_r = cp_cam[0][1], z1_r = cp_cam[0][2];
  double x2_r = cp_cam[1][0], y2_r = cp_cam[1][1], z2_r = cp_cam[1][2];

  cv::Mat A = (cv::Mat_<double>(2, 2) << x1_l * y1_r, -z1_l * y1_r, x2_l * y2_r,
               -z2_l * y2_r);
  cv::Mat B = (cv::Mat_<double>(2, 2) << x1_r * y1_l, z1_r * y1_l, x2_r * y2_l,
               z2_r * y2_l);

  cv::Mat C = B.inv() * A;
  cv::Mat CTC = C.t() * C;

  // Perform SVD
  cv::SVD svd(CTC, cv::SVD::MODIFY_A);
  cv::Mat U = svd.u;
  cv::Mat S = svd.w; // Singular values

  double s1 = S.at<double>(0);
  double s2 = S.at<double>(1);

  // Calculate E from U, y anc C
  auto calcE = [](const cv::Mat &U, const cv::Mat &y,
                  const cv::Mat &C) -> cv::Mat {
    cv::Mat a = U * y;
    cv::Mat b = C * a;
    cv::Mat E = (cv::Mat_<double>(3, 3) << 0, b.at<double>(0), 0,
                 -a.at<double>(0), 0, a.at<double>(1), 0, b.at<double>(1), 0);
    return E;
  };

  matVec essentialMatrices;

  // Two possible solutions when s1 < 1
  if (s1 < 1) {
    for (int i = 0; i < 2; ++i) {
      double y1 = pow(-1, i);
      double y2 = 0;
      cv::Mat y = (cv::Mat_<double>(2, 1) << y1, y2);
      cv::Mat E = calcE(U, y, C);
      essentialMatrices.push_back(E);
    }
  }
  // Two possible solutions when s2 > 1
  else if (s2 > 1) {
    for (int i = 0; i < 2; ++i) {
      double y1 = 0;
      double y2 = pow(-1, i);
      cv::Mat y = (cv::Mat_<double>(2, 1) << y1, y2);
      cv::Mat E = calcE(U, y, C);
      essentialMatrices.push_back(E);
    }
  }
  // Four possible solutions when s1 >= 1 and s2 <= 1
  else {
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        double y1 = pow(-1, i) * sqrt((1 - s2) / (s1 - s2));
        double y2 = pow(-1, j) * sqrt((s1 - 1) / (s1 - s2));
        cv::Mat y = (cv::Mat_<double>(2, 1) << y1, y2);
        cv::Mat E = calcE(U, y, C);
        essentialMatrices.push_back(E);
      }
    }
  }

  return essentialMatrices;
}

cv::Mat findFundam(const pointVec &original_pixels,
                   const pointVec &corresponding_pixels, const cv::Mat &k,
                   const RANSACFundam &ransac) {

  cv::Mat F = ransac.run(original_pixels, corresponding_pixels, k);
  cv::Mat F_normalized = _normalizeF(F);
  return F_normalized;
}

} // namespace PM

#endif // FPSOLVER_H
