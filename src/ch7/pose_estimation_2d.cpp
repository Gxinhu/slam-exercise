#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include <Eigen/Eigen>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <iostream>
#include <vector>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
void pose_estimation(std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2,
    const std::vector<cv::DMatch>& matches, cv::Mat& rotation_matrix, cv::Mat& move_matrix);

cv::Point2d pixel2cam(const cv::Point2d& point, const cv::Mat& intrinsic_matrix);
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
int main(int argc, char** argv) {
    std::string first_file  = "/workspace/slam-exercise/src/ch7/1.png";
    std::string second_file = "/workspace/slam-exercise/src/ch7/2.png";
    cv::Mat img_1 = cv::imread(first_file, cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(second_file, cv::IMREAD_COLOR);
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
    std::array<double, 9> intrinsic_array = {520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1};
    cv::Mat intrinsic_matrix              = cv::Mat(3, 3, CV_64F, intrinsic_array.data());

    // 关键点是按照像素坐标算的，而本质矩阵则是通过归一化平面计算的
    for (const auto m : matches) {
        cv::Point2d point1            = pixel2cam(keypoints_1[m.queryIdx].pt, intrinsic_matrix);
        std::array<double, 3> y1_data = {point1.x, point1.y, 1};
        cv::Point2d point2            = pixel2cam(keypoints_2[m.queryIdx].pt, intrinsic_matrix);
        std::array<double, 3> y2_data = {point2.x, point2.y, 1};
        cv::Mat y1                    = cv::Mat(y1_data.size(), 1, CV_64F, y1_data.data());
        cv::Mat y2                    = cv::Mat(y2_data.size(), 1, CV_64F, y2_data.data());
        cv::Mat d                     = y2.t() * t_x * rotation_matrix * y1;
        fmt::print("epipolar constraint = {}\n", d);
    }

    return 0;
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