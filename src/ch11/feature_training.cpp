#include <DBoW3/BowVector.h>
#include <DBoW3/DBoW3.h>
#include <DBoW3/Database.h>
#include <DBoW3/QueryResults.h>
#include <DBoW3/Vocabulary.h>
#include <cstddef>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
int main() {
    fmt::print("reading image...\n");
    std::vector<cv::Mat> images;
    for (int i = 1; i <= 10; ++i) {
        auto image_path = fmt::format("/workspace/slam-exercise/src/ch11/data/{}.png", i);
        images.push_back(cv::imread(image_path));
    }
    fmt::print("detecting images features\n");
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    std::vector<cv::Mat> descriptors;
    for (auto image : images) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        detector->detectAndCompute(image, cv::Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
    }
    // 采用同一时间采集的特征的生成字典，容易过拟合
    DBoW3::Vocabulary vocab;
    vocab.create(descriptors);
    vocab.save("./src/ch11/vo.yml");
    return 0;
}