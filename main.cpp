#include "fpsolver.h"
#include "ransac.h"
#include <cmath>
#include <iostream>
#include <vector>

struct Point2D {
  float x, y;
};

struct LineModel {
  float a, b; // Line equation: y = ax + b
};

// Estimate a line model from two points
LineModel estimateLineModel(const std::vector<Point2D> &points) {
  LineModel model;
  if (points.size() == 2) {
    float x1 = points[0].x, y1 = points[0].y;
    float x2 = points[1].x, y2 = points[1].y;
    model.a = (y2 - y1) / (x2 - x1);
    model.b = y1 - model.a * x1;
  }
  return model;
}

// Evaluate how well a line model fits a point
float evaluateLineModel(const LineModel &model, const Point2D &point) {
  float y_pred = model.a * point.x + model.b;
  return std::abs(y_pred - point.y);
}

int main() {
  std::vector<Point2D> points = {
      {1, 1},   {2, 2},   {3, 3},   {4, 4}, {5, 5},
      {1, 1.1}, {2, 2.2}, {3, 3.3}, {6, 8} // Outlier at (6, 8)
  };

  RANSAC<Point2D, LineModel> ransac(100, 0.5, 0.99);
  LineModel bestModel =
      ransac.run(points, estimateLineModel, evaluateLineModel);

  std::cout << "Best model: y = " << bestModel.a << "x + " << bestModel.b
            << std::endl;

  return 0;
}
