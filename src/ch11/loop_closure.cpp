#include <DBoW3/BowVector.h>
#include <DBoW3/DBoW3.h>
#include <DBoW3/Database.h>
#include <DBoW3/QueryResults.h>
#include <DBoW3/Vocabulary.h>
#include <cstddef>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/ostream.h>
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
    vocab.load("./src/ch11/vo.yml");
    fmt::print("{}\n",vocab);
    fmt::print("creating vocabulary ...\n");
    fmt::print("done \n");
    fmt::print("compare images with image");
    for (int i = 0; i < images.size(); ++i) {
        DBoW3::BowVector v1;
        vocab.transform(descriptors[i], v1);
        for (int j = i; j < images.size(); ++j) {
            DBoW3::BowVector v2;
            vocab.transform(descriptors[j], v2);
            double score = vocab.score(v1, v2);
            fmt::print("image {} vs image {}: {}\n", i, j, score);
        }
    }
    fmt::print("comparing images with database");
    DBoW3::Database database(vocab, false, 0);
    for (int i = 0; i < descriptors.size(); ++i) {
        database.add(descriptors[i]);
    }
    for (int i = 0; i < descriptors.size(); ++i) {
        DBoW3::QueryResults results;
        database.query(descriptors[i], results, 4);
        fmt::print("searching for image {}\n return {}\n", i, results);
    }
    return 0;
}