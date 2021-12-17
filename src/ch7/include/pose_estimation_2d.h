
#ifndef CH7_POSE_2D_H
#define CH7_POSE_2D_H
#include <Eigen/Eigen>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <iostream>
#include <vector>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

inline std::array<double, 9> intrinsic_array = {520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1};
inline const cv::Mat kIntrinsicMatrix        = cv::Mat(3, 3, CV_64F, intrinsic_array.data());
void pose_estimation(std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2,
    const std::vector<cv::DMatch>& matches, cv::Mat& rotation_matrix, cv::Mat& move_matrix);

inline cv::Scalar get_color(float depth);
cv::Point2d pixel2cam(const cv::Point2d& point, const cv::Mat& intrinsic_matrix);
void triangulation(const std::vector<cv::KeyPoint>& keypoints_1, const std::vector<cv::KeyPoint>& keypoints_2,
    const std::vector<cv::DMatch>& matches, const cv::Mat& rotation_matrix, const cv::Mat& move_matrix,
    const cv::Mat& intrinsic_matrix, std::vector<cv::Point3d>& points);
void find_feature_matches(const cv::Mat& img_1, const cv::Mat& img_2, std::vector<cv::KeyPoint>& keypoints_1,
    std::vector<cv::KeyPoint>& keypoints_2, std::vector<cv::DMatch>& matches);

#endif