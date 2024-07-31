#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Function to draw an "X" on the image
void drawX(Mat &image, Point p, Scalar color, int size = 10) {
  line(image, Point(p.x - size, p.y - size), Point(p.x + size, p.y + size),
       color, 2);
  line(image, Point(p.x - size, p.y + size), Point(p.x + size, p.y - size),
       color, 2);
}

int main() {
  // Path to your image
  string imagePath = "/home/batman/fun/pose-from-planar-motion/input_images/"
                     "1.jpg";

  // Load the image
  Mat img = imread(imagePath);
  if (img.empty()) {
    cout << "Could not open or find the image." << endl;
    return -1;
  }

  // Coordinates where you want to draw the "X"
  int x = 1122;                               // Replace with your x coordinate
  int y = 361;                                // Replace with your y coordinate
  drawX(img, Point(x, y), Scalar(0, 0, 255)); // Red color for the "X"

  int x2 = 1086; // Replace with your x coordinate
  int y2 = 408;  // Replace with your y coordinate
  drawX(img, Point(x2, y2), Scalar(0, 0, 255)); // Red color for the "X"

  // Display the image
  namedWindow("Image", WINDOW_AUTOSIZE);
  imshow("Image", img);

  // Wait indefinitely until a key is pressed
  waitKey(0);
  cv::destroyAllWindows();

  string imagePath2 = "/home/batman/fun/pose-from-planar-motion/input_images/"
                      "2.jpg";
  Mat img2 = imread(imagePath2);
  // Coordinates where you want to draw the "X"
  x = 997;                                     // Replace with your x coordinate
  y = 270;                                     // Replace with your y coordinate
  drawX(img2, Point(x, y), Scalar(0, 0, 255)); // Red color for the "X"

  x2 = 946; // Replace with your x coordinate
  y2 = 342; // Replace with your y coordinate
  drawX(img2, Point(x2, y2), Scalar(0, 0, 255)); // Red color for the "X"

  // Display the image
  namedWindow("Image2", WINDOW_AUTOSIZE);
  imshow("Image2", img2);

  // Wait indefinitely until a key is pressed
  waitKey(0);
  return 0;
}
