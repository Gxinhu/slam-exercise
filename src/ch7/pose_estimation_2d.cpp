#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include "include/pose_estimation_2d.h"

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

int main1() {
    std::string first_file  = "/workspace/slam-exercise/src/ch7/1.png";
    std::string second_file = "/workspace/slam-exercise/src/ch7/2.png";
    cv::Mat img_1           = cv::imread(first_file, cv::IMREAD_COLOR);
    cv::Mat img_2           = cv::imread(second_file, cv::IMREAD_COLOR);
    assert(img_1.data && img_2.data && "Can not load images!");

    std::vector<cv::KeyPoint> keypoints_1;
    std::vector<cv::KeyPoint> keypoints_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    std::cout << "一共找到了" << matches.size() << "组匹配点" << std::endl;
    cv::Mat rotation_matrix;
    cv::Mat move_matrix;
    pose_estimation(keypoints_1, keypoints_2, matches, rotation_matrix, move_matrix);
    std::array<double, 9> data = {0, -move_matrix.at<double>(2, 0), move_matrix.at<double>(1, 0),
        move_matrix.at<double>(2, 0), 0, -move_matrix.at<double>(0, 0), -move_matrix.at<double>(1, 0),
        move_matrix.at<double>(0, 0), 0};
    cv::Mat t_x                = cv::Mat(3, 3, CV_64F, data.data());
    fmt::print("t^*R={}\n", t_x * rotation_matrix);

    // 关键点是按照像素坐标算的，而本质矩阵则是通过归一化平面计算的
    for (const auto m : matches) {
        cv::Point2d point1            = pixel2cam(keypoints_1[m.queryIdx].pt, kIntrinsicMatrix);
        std::array<double, 3> y1_data = {point1.x, point1.y, 1};
        cv::Point2d point2            = pixel2cam(keypoints_2[m.queryIdx].pt, kIntrinsicMatrix);
        std::array<double, 3> y2_data = {point2.x, point2.y, 1};
        cv::Mat y1                    = cv::Mat(y1_data.size(), 1, CV_64F, y1_data.data());
        cv::Mat y2                    = cv::Mat(y2_data.size(), 1, CV_64F, y2_data.data());
        cv::Mat d                     = y2.t() * t_x * rotation_matrix * y1;
        fmt::print("epipolar constraint = {}\n", d);
    }

    std::vector<cv::Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, rotation_matrix, move_matrix, kIntrinsicMatrix, points);

    cv::Mat img1_plot = img_1.clone();
    cv::Mat img2_plot = img_2.clone();
    for (int i = 0; i < matches.size(); i++) {
        // 第一个图
        float depth1 = points[i].z;
        std::cout << "depth: " << depth1 << std::endl;
        cv::Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, kIntrinsicMatrix);
        cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);

        // 第二个图
        std::array<double, 3> point_data = {points[i].x, points[i].y, points[i].z};
        cv::Mat pt2_trans                = kIntrinsicMatrix * (cv::Mat(3, 1, CV_64F, point_data.data())) + move_matrix;
        float depth2                     = pt2_trans.at<double>(2, 0);
        cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
    }
    cv::imshow("img 1", img1_plot);
    cv::imshow("img 2", img2_plot);
    cv::waitKey();
    return 0;
}
inline cv::Scalar get_color(float depth) {
    float up_th = 50, low_th = 10, th_range = up_th - low_th;
    if (depth > up_th)
        depth = up_th;
    if (depth < low_th)
        depth = low_th;
    return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}
cv::Point2d pixel2cam(const cv::Point2d& point, const cv::Mat& intrinsic_matrix) {

    return {(point.x - intrinsic_matrix.at<double>(0, 2)) / intrinsic_matrix.at<double>(0, 0),
        (point.y - intrinsic_matrix.at<double>(1, 2)) / intrinsic_matrix.at<double>(1, 1)};
}

void pose_estimation(std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2,
    const std::vector<cv::DMatch>& matches, cv::Mat& rotation_matrix, cv::Mat& move_matrix) {
    std::vector<cv::Point2d> points1;
    std::vector<cv::Point2d> points2;
    for (const auto& match : matches) {
        points1.push_back(keypoints_1[match.queryIdx].pt);
        points2.push_back(keypoints_2[match.queryIdx].pt);
    }

    cv::Mat fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
    fmt::print("fundamental matrix is : {}\n", fundamental_matrix);


    // 相机光心
    const cv::Point2d principal_point(325.1, 249.7);
    // 相机焦距
    const double focal_length = 521;
    cv::Mat essential_matrix  = cv::findEssentialMat(points1, points2, focal_length, principal_point);
    fmt::print("essential matrix is : {}\n", essential_matrix);

    cv::Mat homograph_matrix = cv::findHomography(points1, points2, cv::RANSAC, 3);
    fmt::print("homograph matrix is {} \n", homograph_matrix);

    cv::recoverPose(essential_matrix, points1, points2, rotation_matrix, move_matrix, focal_length, principal_point);
    fmt::print("rotation matrix is : {} \n", rotation_matrix);
    fmt::print("move matrix is : {}\n", move_matrix);
}

void triangulation(const std::vector<cv::KeyPoint>& keypoints_1, const std::vector<cv::KeyPoint>& keypoints_2,
    const std::vector<cv::DMatch>& matches, const cv::Mat& rotation_matrix, const cv::Mat& move_matrix,
    const cv::Mat& intrinsic_matrix, std::vector<cv::Point3d>& points) {
    std::vector<cv::Point2f> camera_points1;
    std::vector<cv::Point2f> camera_points2;
    camera_points1.reserve(matches.size());
    for (const auto& m : matches) {
        camera_points1.push_back(pixel2cam(keypoints_1[m.queryIdx].pt, intrinsic_matrix));
        camera_points2.push_back(pixel2cam(keypoints_2[m.queryIdx].pt, intrinsic_matrix));
    }
    std::array<float, 12> no_trans_data = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    cv::Mat matrices[]                  = {rotation_matrix, move_matrix};

    std::vector<double> t2  = {rotation_matrix.at<double>(0, 0), rotation_matrix.at<double>(0, 1),
        rotation_matrix.at<double>(0, 2), move_matrix.at<double>(0, 0), rotation_matrix.at<double>(1, 0),
        rotation_matrix.at<double>(1, 1), rotation_matrix.at<double>(1, 2), move_matrix.at<double>(1, 0),
        rotation_matrix.at<double>(2, 0), rotation_matrix.at<double>(2, 1), rotation_matrix.at<double>(2, 2),
        move_matrix.at<double>(2, 0)};
    cv::Mat no_trans_matrix = cv::Mat(3, 4, CV_32F, no_trans_data.data());
    cv::Mat points_4d;
    cv::Mat translation_matrix = cv::Mat(3, 4, CV_32F, t2.data());
    // 前两个矩阵就是两个坐标系的转化方式
    // opencv 4中深度信息计算有问题,但是现在一般不是使用单目，暂时跳过
    cv::triangulatePoints(no_trans_matrix, translation_matrix, camera_points1, camera_points2, points_4d);

    // 转化为非齐次座标
    for (int i = 0; i < points_4d.cols; ++i) {
        cv::Mat x = points_4d.col(i);
        x /= x.at<float>(3, 0);
        cv::Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        points.push_back(p);
    }
}

void find_feature_matches(const cv::Mat& img_1, const cv::Mat& img_2, std::vector<cv::KeyPoint>& keypoints_1,
    std::vector<cv::KeyPoint>& keypoints_2, std::vector<cv::DMatch>& matches) {
    //-- 初始化
    cv::Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    cv::Ptr<cv::FeatureDetector> detector       = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    std::vector<cv::DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) {
            min_dist = dist;
        }
        if (dist > max_dist) {
            max_dist = dist;
        }
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= std::max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}