// 使用三角函数的原理可以回避 arctan、cos、sin 的计算
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
using DescType = std::vector<uint32_t>;
void ComputeOrb(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, std::vector<DescType>& descriptors) {

    const int half_patch_size = 8;
    const int half_boundary   = 16;
    int bad_point_number      = 0;

    for (auto& keypoint : keypoints) {
        auto is_top_boundary = keypoint.pt.x < half_boundary || keypoint.pt.y < half_boundary;
        auto is_bottom_boundary =
            keypoint.pt.x >= img.cols - half_boundary || keypoint.pt.y >= img.rows - half_boundary;
        auto is_boundary = is_top_boundary || is_bottom_boundary;
        if (is_boundary) {
            bad_point_number++;
            descriptors.emplace_back();
            continue;
        }
        float m01 = 0;
        float m10 = 0;
        for (int dx = -half_patch_size; dx < half_patch_size; ++dx) {
            for (int dy = -half_patch_size; dy < half_patch_size; ++dy) {
                // uchar [0-255]
                uchar pixel = img.at<uchar>(keypoint.pt.y + dy, keypoint.pt.x + dx);
                // 这里的 m01、10 的顺序我有点懵
                m01 += dx * pixel;
                m10 += dy * pixel;
            }
            cv::Moments me();
        }
    }
}
