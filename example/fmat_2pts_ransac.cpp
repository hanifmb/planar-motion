#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Mouse callback function to show pixel location
void onMouse(int event, int x, int y, int, void *) {
  if (event == EVENT_LBUTTONDOWN) {
    cout << "Clicked at: (" << x << ", " << y << ")" << endl;
  }
}

int main() {
  // Hardcoded image path
  string imagePath = "/home/batman/fun/pose-from-planar-motion/input_images/"
                     "2.jpg";

  // Load the image
  Mat img = imread(imagePath);
  if (img.empty()) {
    cout << "Could not open or find the image." << endl;
    return -1;
  }

  // Display the image
  namedWindow("Image", WINDOW_AUTOSIZE);
  imshow("Image", img);

  // Set the mouse callback function
  setMouseCallback("Image", onMouse);

  // Wait indefinitely until a key is pressed
  waitKey(0);

  return 0;
}
