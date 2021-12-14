#include <chrono>
#include <cmath>
#include <fmt/core.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "usage: features extraction img1 img2" << std::endl;
        return 1;
    }
    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_COLOR);
    assert(img1.data != nullptr && img2.data != nullptr);

    std::vector<cv::KeyPoint> keypoints_1;
    std::vector<cv::KeyPoint> keypoints_2;
    cv::Mat descriptors_1;
    cv::Mat descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector        = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptors = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher       = cv::DescriptorMatcher::create("BruteForce-Hamming");

    auto t1 = std::chrono::steady_clock::now();
    detector->detect(img1, keypoints_1);
    detector->detect(img2, keypoints_2);

    descriptors->compute(img1, keypoints_1, descriptors_1);
    descriptors->compute(img2, keypoints_2, descriptors_2);
    auto t2 = std::chrono::steady_clock::now();

    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    fmt::print("extract ORB cost = {} seconds\n", time_used.count());

    cv::Mat outimg_1;
    cv::drawKeypoints(img1, keypoints_1, outimg_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB features", outimg_1);

    std::vector<cv::DMatch> matches;
    t1 = std::chrono::steady_clock::now();
    matcher->match(descriptors_1, descriptors_2, matches);
    t2 = std::chrono::steady_clock::now();

    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    fmt::print("match ORB cost = {} seconds\n", time_used.count());

    auto minmax = minmax_element(matches.begin(), matches.end(),
        [](const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; });

    double min_dist = minmax.first->distance;
    double max_dist = minmax.second->distance;
    fmt::print("Max distance : {}\n", max_dist);
    fmt::print("Min distance : {}\n", min_dist);

    // 30.0 是为了描述子之间的最小距离过于小，所采取的一个经验值
    // 未经筛选的匹配具有大量的误匹配，所以还要选择出来好的匹配。
    const double empirical_value_min_dist = 30.0;
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= std::max(2 * min_dist, empirical_value_min_dist)) {
            good_matches.push_back(matches[i]);
        }
    }
    fmt::print(" the number of all matches : {}\n",matches.size());
    fmt::print(" the number of good matches : {}\n",good_matches.size());

    cv::Mat img_match;
    cv::Mat img_goodmatch;
    cv::drawMatches(img1, keypoints_1, img2, keypoints_2, matches, img_match);
    cv::drawMatches(img1, keypoints_1, img2, keypoints_2, good_matches, img_goodmatch);
    cv::imshow("all matchs", img_match);
    cv::imshow("good matches", img_goodmatch);
    cv::waitKey(0);
    return 0;
}