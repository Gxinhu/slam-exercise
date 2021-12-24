#include <Eigen/Eigen>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fmt/core.h>
#include <functional>
#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
class OpticalFlowTracker {
public:
    OpticalFlowTracker(const cv::Mat& img1, const cv::Mat& img2,
        const std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
        std::vector<bool>& success, bool inverse = true, bool has_initial = false)
        : img1_(img1), img2_(img2), keypoints1_(kp1), keypoints2_(kp2), success_(success),
          inverse_(inverse), has_initial_(has_initial){};
    void calculateOpticalFlow(const cv::Range& range);

private:
    const cv::Mat& img1_;
    const cv::Mat& img2_;
    const std::vector<cv::KeyPoint>& keypoints1_;
    std::vector<cv::KeyPoint>& keypoints2_;
    std::vector<bool>& success_;
    bool inverse_     = true;
    bool has_initial_ = false;
};

inline float GetPixelValue(const cv::Mat& img, float x, float y);
void OpticalFlowSingleLevel(const cv::Mat& img1, const cv::Mat& img2,
    const std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
    std::vector<bool>& success, bool inverse = false, bool has_initial_guess = false);
void OpticalFlowMultiLevel(const cv::Mat& img1, const cv::Mat& img2,
    const std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
    std::vector<bool>& success, bool inverse);
int main(int argc, char** argv) {
    std::string file1 = "/workspace/slam-exercise/src/ch8/LK1.png";
    std::string file2 = "/workspace/slam-exercise/src/ch8/LK2.png";
    cv::Mat img1      = cv::imread(file1, CV_8UC1);
    cv::Mat img2      = cv::imread(file2, CV_8UC1);
    assert(img1.data != nullptr && img2.data != nullptr);

    std::vector<cv::KeyPoint> keypoints_1;
    cv::Ptr<cv::FeatureDetector> detector = cv::GFTTDetector::create(500, 0.01, 20);
    auto t1                               = std::chrono::steady_clock::now();
    detector->detect(img1, keypoints_1);
    std::vector<uchar> status;
    std::vector<float> error;
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    for (auto keypoint : keypoints_1) {
        points1.push_back(keypoint.pt);
    }
    cv::calcOpticalFlowPyrLK(img1, img2, points1, points2, status, error);
    auto t2        = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    fmt::print("extract ORB cost = {} seconds\n", time_used.count());

    std::vector<cv::KeyPoint> keypoint_single;
    std::vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, keypoints_1, keypoint_single, success_single);
    std::vector<cv::KeyPoint> keypoint_multi;
    std::vector<bool> success_multi;
    OpticalFlowMultiLevel(img1, img2, keypoints_1, keypoint_multi, success_multi, true);


    // plot the differences of those functions
    cv::Mat img2_single;
    cv::cvtColor(img2, img2_single, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < keypoint_single.size(); i++) {
        if (success_single[i]) {
            cv::circle(img2_single, keypoint_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, keypoints_1[i].pt, keypoint_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    cv::Mat img2_multi;
    cv::cvtColor(img2, img2_multi, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < keypoint_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, keypoint_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, keypoints_1[i].pt, keypoint_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    cv::Mat img2_CV;
    cv::cvtColor(img2, img2_CV, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < points2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, points2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, points1[i], points2[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    cv::waitKey(0);

    return 0;
}


void OpticalFlowSingleLevel(const cv::Mat& img1, const cv::Mat& img2,
    const std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
    std::vector<bool>& success, bool inverse, bool has_initial) {
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    cv::parallel_for_(cv::Range(0, kp1.size()),
        std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, std::placeholders::_1));
}


void OpticalFlowTracker::calculateOpticalFlow(const cv::Range& range) {
    const int half_patch_size = 4;
    const int iterations      = 10;
    for (size_t i = range.start; i < range.end; ++i) {
        auto kp   = keypoints1_[i];
        double dx = 0;
        double dy = 0;
        if (has_initial_) {
            dx = keypoints2_[i].pt.x - kp.pt.x;
            dy = keypoints2_[i].pt.y - kp.pt.y;
        }
        double cost      = 0;
        double last_cost = 0;
        bool succ        = true;

        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        Eigen::Vector2d jacobian;

        for (int iter = 0; iter < iterations; ++iter) {
            if (inverse_ == false) {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } else {
                b = Eigen::Vector2d::Zero();
            }
            cost = 0;
            for (int x = -half_patch_size; x < half_patch_size; ++x) {
                for (int y = -half_patch_size; y < half_patch_size; ++y) {
                    double error = GetPixelValue(img1_, kp.pt.x + x, kp.pt.y + y)
                                 - GetPixelValue(img2_, kp.pt.x + x + dx, kp.pt.y + dy + y);
                    if (inverse_ == false) {
                        jacobian =
                            -1.00
                            * Eigen::Vector2d(
                                0.5
                                    * (GetPixelValue(img2_, kp.pt.x + dx + x + 1, kp.pt.y + y + dy)
                                        - GetPixelValue(
                                            img2_, x + kp.pt.x + dx - 1, y + kp.pt.y + dy)),
                                0.5
                                    * (GetPixelValue(img2_, kp.pt.x + dx + x, kp.pt.y + y + dy + 1)
                                        - GetPixelValue(
                                            img2_, x + kp.pt.x + dx, y + kp.pt.y + dy - 1)));
                    } else if (iter == 0) {

                        jacobian =
                            -1.00
                            * Eigen::Vector2d(
                                0.5
                                    * (GetPixelValue(img1_, kp.pt.x + dx + x + 1, kp.pt.y + y + dy)
                                        - GetPixelValue(
                                            img1_, x + kp.pt.x + dx - 1, y + kp.pt.y + dy)),
                                0.5
                                    * (GetPixelValue(img1_, kp.pt.x + dx + x, kp.pt.y + y + dy + 1)
                                        - GetPixelValue(
                                            img1_, x + kp.pt.x + dx, y + kp.pt.y + dy - 1)));
                    }
                    b += -error * jacobian;
                    cost += error * error;
                    if (inverse_ == false || iter == 0) {
                        H += jacobian * jacobian.transpose();
                    }
                }
            }
            Eigen::Vector2d update = H.ldlt().solve(b);
            if (std::isnan(update[0])) {
                std::cerr << "update is nan" << std::endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > last_cost) {
                break;
            }
            dx += update[0];
            dy += update[1];
            last_cost = cost;
            succ      = true;

            if (update.norm() < 1e-2) {
                break;
            }
        }
        success_[i]       = succ;
        keypoints2_[i].pt = kp.pt + cv::Point2f(dx, dy);
    }
}

inline float GetPixelValue(const cv::Mat& img, float x, float y) {
    // boundary check
    if (x < 0)
        x = 0;
    if (y < 0)
        y = 0;
    if (x >= img.cols - 1)
        x = img.cols - 2;
    if (y >= img.rows - 1)
        y = img.rows - 2;

    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);

    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x) + xx * (1 - yy) * img.at<uchar>(y, x_a1)
         + (1 - xx) * yy * img.at<uchar>(y_a1, x) + xx * yy * img.at<uchar>(y_a1, x_a1);
}
void OpticalFlowMultiLevel(const cv::Mat& img1, const cv::Mat& img2,
    const std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
    std::vector<bool>& success, bool inverse) {
    const int pyramids         = 4;
    const double pyramid_scale = 0.5;
    const double scales[]      = {1.0, 0.5, 0.25, 0.125};
    std::vector<cv::Mat> pyramid1, pyramid2;
    for (int i = 0; i < pyramids; ++i) {
        if (i == 0) {
            pyramid1.push_back(img1);
            pyramid2.push_back(img2);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyramid1[i - 1], img1_pyr,
                cv::Size(
                    pyramid1[i - 1].cols * pyramid_scale, pyramid1[i - 1].rows * pyramid_scale));
            cv::resize(pyramid2[i - 1], img2_pyr,
                cv::Size(
                    pyramid2[i - 1].cols * pyramid_scale, pyramid2[i - 1].rows * pyramid_scale));
            pyramid1.push_back(img1_pyr);
            pyramid2.push_back(img2_pyr);
        }
    }
    std::vector<cv::KeyPoint> kp1_pyr, kp2_pyr;
    for (auto& kp : kp1) {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    for (int level = pyramids - 1; level >= 0; level--) {
        success.clear();
        OpticalFlowSingleLevel(
            pyramid1[level], pyramid2[level], kp1_pyr, kp2_pyr, success, inverse, true);
        if (level > 0) {
            for (auto& kp : kp1_pyr) {
                kp.pt /= pyramid_scale;
            }
            for (auto& kp : kp2_pyr) {
                kp.pt /= pyramid_scale;
            }
        }
    }

    for (auto kp : kp2_pyr) {
        kp2.push_back(kp);
    }
}
