#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
const std::string IMAGE_FILE = "./distorted.png";

int main(int argc, char** argv) {
    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;
    cv::Mat image = cv::imread(IMAGE_FILE, 0);
    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistorted = cv::Mat(rows, cols, CV_8UC1);
    int count = 0;
    int count2 = 0;
    for (int u = 0; u < cols; ++u) {
        for (int v = 0; v < rows; ++v) {
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double r = sqrt(pow(x, 2) + pow(y, 2));
            double x_undistorted = x * (1 + k1 * pow(r, 2) + k2 * pow(r, 4)) + 2 * p1 * x * y + p2 * (pow(r, 2) + 2 * pow(x, 2));
            double y_undistorted = y * (1 + k1 * pow(r, 2) + k2 * pow(r, 4)) + 2 * p2 * x * y + p1 * (pow(r, 2) + 2 * pow(y, 2));
            double u_undistorted = x_undistorted * fx + cx;
            double v_undistorted = y_undistorted * fy + cy;
            if (u_undistorted >= 0 && v_undistorted >= 0 && u_undistorted < cols && v_undistorted < rows) {
                image_undistorted.at<uchar>(v, u) = image.at<uchar>((int)v_undistorted, (int)u_undistorted);
                count++;
            } else {
                image_undistorted.at<uchar>(v, u) = 0;
                count2++;
            }
        }
    }
    cv::imshow("image", image);
    cv::imshow("image_distorted", image_undistorted);
    cv::waitKey(0);
    std::cout << count << " " << count2 << std::endl;
    return 0;
}