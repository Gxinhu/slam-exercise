#include <fmt/core.h>

#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv) {
    cv::Mat image;
    image = cv::imread("/workspace/slam-exercise/xx.png");

    if (image.data == nullptr) {
        std::cerr << " file " << argv[1] << " not exist!" << std::endl;
    }
    std::cout << " width = " << image.cols << " height = " << image.rows
              << " channel = " << image.channels() << std::endl;
    std::cout << "Image type is " << image.type() << std::endl;

    if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {
        std::cout << " please input a image with color or grey" << std::endl;
        return 0;
    }

    std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();
    for (size_t y = 0; y < image.rows; ++y) {
        auto row_ptr = image.ptr<unsigned int>(y);
        for (size_t x = 0; x < image.cols; ++x) {
            auto value = &row_ptr[x * image.channels()];
            for (int c = 0; c != image.channels(); ++c) {
                auto data = value[c];
            }
        }
    }
    std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();

    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
            for (int c = 0; c != image.channels(); ++c) {
                auto value = image.at<cv::Vec3d>(i, j)[c];
            }
    // You can now access the pixel value with cv::Vec3b
    std::chrono::steady_clock::time_point time3 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_duration1 = std::chrono::duration_cast<std::chrono::duration<double>>(time2 - time1);
    std::chrono::duration<double> time_duration2 = std::chrono::duration_cast<std::chrono::duration<double>>(time3 - time2);
    fmt::print("iteration time1:{}, iteration time2: {}\n", time_duration1.count(), time_duration2.count());
    // 深浅拷贝问题
    cv::Mat image_another = image;
    image_another(cv::Rect(0, 0, 100, 100)).setTo(0);
    cv::waitKey(0);
    cv::Mat image_clone = image.clone();
    image_another(cv::Rect(0, 0, 100, 100)).setTo(255);
    cv::imshow("Image_clone", image_clone);
    cv::imshow("Image", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}