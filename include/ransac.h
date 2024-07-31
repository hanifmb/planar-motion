#ifndef RANSAC_H
#define RANSAC_H

#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <vector>

template <typename T, typename Model> class RANSAC {
public:
  using DataPoint = T;
  using ModelType = Model;

  RANSAC(int maxIterations, float distanceThreshold, float confidence)
      : maxIterations(maxIterations), distanceThreshold(distanceThreshold),
        confidence(confidence) {
    rng.seed(std::random_device{}());
  }

  ModelType
  run(const std::vector<DataPoint> &dataPoints,
      std::function<ModelType(const std::vector<DataPoint> &)> modelEstimator,
      std::function<float(const ModelType &, const DataPoint &)>
          modelEvaluator) {
    int bestInlierCount = 0;
    ModelType bestModel;

    int numDataPoints = dataPoints.size();
    for (int iteration = 0; iteration < maxIterations; ++iteration) {
      // Randomly sample data points
      std::vector<DataPoint> sample = sampleRandomPoints(dataPoints);

      // Estimate model
      ModelType model = modelEstimator(sample);

      // Evaluate model
      int inlierCount = 0;
      for (const auto &point : dataPoints) {
        float distance = modelEvaluator(model, point);
        if (distance < distanceThreshold) {
          ++inlierCount;
        }
      }

      // Update best model if this one is better
      if (inlierCount > bestInlierCount) {
        bestInlierCount = inlierCount;
        bestModel = model;
      }

      // Check if we can terminate early
      float inlierRatio = static_cast<float>(bestInlierCount) / numDataPoints;
      float threshold = 1 - pow(1 - inlierRatio, maxIterations);
      if (threshold > confidence) {
        break;
      }
    }

    return bestModel;
  }

private:
  int maxIterations;
  float distanceThreshold;
  float confidence;
  std::mt19937 rng;

  std::vector<DataPoint>
  sampleRandomPoints(const std::vector<DataPoint> &dataPoints) {
    std::vector<DataPoint> sample;
    std::uniform_int_distribution<> dist(0, dataPoints.size() - 1);
    for (size_t i = 0; i < 2; ++i) { // Assuming we need 2 points for the model
      sample.push_back(dataPoints[dist(rng)]);
    }
    return sample;
  }
};

#endif // RANSAC_H
