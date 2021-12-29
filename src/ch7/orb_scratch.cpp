// 使用三角函数的原理可以回避 arctan、cos、sin 的计算
#include <chrono>
#include <cmath>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <nmmintrin.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <popcntintrin.h>
#include <string>
#include <yaml-cpp/yaml.h>

const int kNumberUint = 8;
const int kUintBits   = 32;

// 32 位的 unsigned int 数据
using DescType = std::vector<uint32_t>;

void ComputeOrb(
    const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, std::vector<DescType>& descriptors);
void BruteForceMatch(const std::vector<DescType>& desc1, const std::vector<DescType>& desc2,
    std::vector<cv::DMatch>& matches);

int main() {

    // load image
    std::string first_file  = "/workspace/slam-exercise/src/ch7/1.png";
    std::string second_file = "/workspace/slam-exercise/src/ch7/2.png";
    cv::Mat first_image     = cv::imread(first_file, 0);
    cv::Mat second_image    = cv::imread(second_file, 0);
    assert(first_image.data != nullptr && second_image.data != nullptr);

    // detect FAST keypoints1 using threshold=40
    auto t1 = std::chrono::steady_clock::now();
    std::vector<cv::KeyPoint> keypoints1;
    cv::FAST(first_image, keypoints1, 40);
    std::vector<DescType> descriptor1;
    ComputeOrb(first_image, keypoints1, descriptor1);

    // same for the second
    std::vector<cv::KeyPoint> keypoints2;
    std::vector<DescType> descriptor2;
    cv::FAST(second_image, keypoints2, 40);
    ComputeOrb(second_image, keypoints2, descriptor2);
    auto t2        = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "extract ORB cost = " << time_used.count() << " seconds. " << std::endl;

    // find matches
    std::vector<cv::DMatch> matches;
    t1 = std::chrono::steady_clock::now();
    BruteForceMatch(descriptor1, descriptor2, matches);
    t2        = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "match ORB cost = " << time_used.count() << " seconds. " << std::endl;
    std::cout << "matches: " << matches.size() << std::endl;

    // plot the matches
    cv::Mat image_show;
    cv::drawMatches(first_image, keypoints1, second_image, keypoints2, matches, image_show);
    cv::imshow("matches", image_show);
    cv::imwrite("matches.png", image_show);
    cv::waitKey(0);

    std::cout << "done." << std::endl;
    return 0;
}
auto ReadMatFromYaml(const std::string& filename, cv::Mat& orb_pattern) {
    YAML::Node orb_array = YAML::LoadFile(filename);
    for (int cnt = 0; cnt < orb_array.size(); ++cnt) {
        int temprow                           = cnt / orb_pattern.cols;
        int tempcol                           = cnt % orb_pattern.cols;
        orb_pattern.at<int>(temprow, tempcol) = orb_array[cnt].as<int>();
    }
}
void ComputeOrb(
    const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, std::vector<DescType>& descriptors) {
    cv::Mat orb_pattern = cv::Mat::zeros(256, 4, CV_64FC1);
    ReadMatFromYaml("/workspace/slam-exercise/src/ch7/orb_pattern.yaml", orb_pattern);
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
                m10 += dx * pixel;
                m01 += dy * pixel;
            }
        }

        auto m_sqrt     = sqrt(m01 * m01 + m10 * m10) + 1e-18;
        float sin_theta = m01 / m_sqrt;
        float cos_theta = m10 / m_sqrt;
        DescType desc(kNumberUint, 0);
        for (int i = 0; i < kNumberUint; ++i) {
            uint32_t d = 0;
            for (int k = 0; k < kUintBits; ++k) {
                int idx_pq = i * kUintBits + k;
                // 采用 ORB 模型选取采样点来对比 Brief
                cv::Point2f p(orb_pattern.at<int>(idx_pq, 0), orb_pattern.at<int>(idx_pq, 1));
                cv::Point2f q(orb_pattern.at<int>(idx_pq, 2), orb_pattern.at<int>(idx_pq, 3));
                // 利用先前特征点的方向信息，计算旋转后的特征，可以使得 ORB
                // 描述子有着更好的，旋转不变性
                cv::Point2f pp = cv::Point2f(cos_theta * p.x - sin_theta * p.y,
                                     sin_theta * p.x + cos_theta * p.y)
                               + keypoint.pt;
                cv::Point2f qq = cv::Point2f(cos_theta * q.x - sin_theta * q.y,
                                     sin_theta * q.x + cos_theta * q.y)
                               + keypoint.pt;
                if (img.at<uchar>(pp.y, pp.x) < img.at<uchar>(qq.y, qq.x)) {
                    // 二进制输入
                    d |= 1 << k;
                }
            }
            desc[i] = d;
        }
        descriptors.push_back(desc);
    }
    fmt::print("bad/total:{}/{}\n", bad_point_number, keypoints.size());
}

//暴力匹配
void BruteForceMatch(const std::vector<DescType>& desc1, const std::vector<DescType>& desc2,
    std::vector<cv::DMatch>& matches) {
    const int d_max = 40;
    for (int i = 0; i < desc1.size(); ++i) {
        if (desc1[i].empty()) {
            continue;
        }
        cv::DMatch m{i, 0, kNumberUint * kUintBits};
        for (int j = 0; j < desc2.size(); ++j) {
            if (desc2[j].empty()) {
                continue;
            }
            int distance = 0;
            for (int k = 0; k < kNumberUint; k++) {
                distance += _mm_popcnt_u32(desc1[i][k] ^ desc2[j][k]);
            }
            if (distance < d_max && distance < m.distance) {
                m.distance = distance;
                m.trainIdx = j;
            }
        }
        if (m.distance < d_max) {
            matches.push_back(m);
        }
    }
}
