# planar-motion
This repository contains C++ examples of fundamental matrix estimation using two-point correspondences under planar motion. It is the implementation of the paper by Chou et. al. cited in the reference below.

#### What is planar motion?

Any agent moving on a planar surface can be described as planar motion. For example, this includes a car moving along a flat road or a multirotor flying at a constant altitude.

#### Why should I care about minimal solutions?

It is particularly useful for robustification against outliers (e.g. in RANSAC). In contrast to the eight-point algorithm, this method requires only two-point correspondences by enforcing a planar motion constraint. This means less iteration to find the best model. 

#### What do we calculate exactly?

Imagine a vehicle in planar motion: its body rotates only along the yaw axis (y-axis). Conversely, there is no translation along the y-axis, so this movement can be ignored.

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?$%5Cmathbf%7BR%7D%5E%7B%5Cprime%7D=%5Cleft%5B%5Cbegin%7Barray%7D%7Bccc%7D%5Ccos%5Ctheta&0&%5Csin%5Ctheta%5C%5C0&1&0%5C%5C-%5Csin%5Ctheta&0&%5Ccos%5Ctheta%5Cend%7Barray%7D%5Cright%5D%5Cquad$and$%5Cquad%5Cmathbf%7Bt%7D%5E%7B%5Cprime%7D=%5Crho%5Cleft%5B%5Cbegin%7Barray%7D%7Bc%7D%5Csin%5Cphi%5C%5C0%5C%5C%5Ccos%5Cphi%5Cend%7Barray%7D%5Cright%5D$" alt="equation">
</p>

After converting the above translation to the essential matrix translation component, both terms can be multiplied to get the essential matrix.

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?$%5Cmathrm%7BE%7D=%5Crho%5Cleft%5B%5Cbegin%7Barray%7D%7Bccc%7D0&%5Ccos(%5Ctheta-%5Cphi)&0%5C%5C-%5Ccos%5Cphi&0&%5Csin%5Cphi%5C%5C0&%5Csin(%5Ctheta-%5Cphi)&0%5Cend%7Barray%7D%5Cright%5D$" alt="equation">
</p>

Solving the matrix above reduces to a geometric problem of finding the intersection between the ellipse and the circle. Assuming the intrinsic matrix is known, the fundamental matrix and its pose can be derived.

## Build

OpenCV is required to build the examples.

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

The simplest example is to calculate the fundamental matrix directly from two visual correspondences. 

```cpp
#include "fpsolver.h"

using pointsVec = std::vector<cv::Point2f>;

int main() {
  // needs to be exactly two correspondences
  pointsVec points1_est = {cv::Point2f(1652, 469), cv::Point2f(1642, 656)};
  pointsVec points2_est = {cv::Point2f(1712, 453), cv::Point2f(1739, 696)};

  cv::Mat k = (cv::Mat_<double>(3, 3) << 1280.7, 0.0, 969.4257, 0.0, 1281.2,
               639.7227, 0.0, 0.0, 1.0); // intrinsic matrix example

  cv::Mat F = PM::findFundam(points1_est, points2_est, k);
}

```

A more useful example is to utilize RANSAC for robustification against outliers.

```cpp
  // requires >= 2 correspondences via feature matching
  pointsVec points1_est = { ... };
  pointsVec points2_est = { ... };

  int max_iter = 1000;
  double threshold = 20;
  double confidence = 99.0;
  PM::RANSACFundam ransac(max_iter, threshold, confidence);

  cv::Mat F = PM::findFundam(points1_est, points2_est, k, ransac);
```


## References

Sunglok Choi, Jong-Hwan Kim,
Fast and reliable minimal relative pose estimation under planar motion,
Image and Vision Computing,
Volume 69,
2018.
