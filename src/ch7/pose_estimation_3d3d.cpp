#include "include/pose_estimation_2d.h"
#include <Eigen/src/Core/Array.h>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/Core/util/Memory.h>
#include <chrono>
#include <cmath>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <g2o/core/base_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/stuff/misc.h>
#include <iostream>
#include <istream>
#include <memory>
#include <opencv2/core/hal/interface.h>
#include <ostream>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sophus/se3.hpp>
using VecVector3d = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
using VecVector2d = std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>;
// 顶点: 需要优化的参数，在 PnP中也就是位姿
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    void oplusImpl(const double* update) override {
        Eigen::Matrix<double, 6, 1> update_vector;
        update_vector << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_vector) * _estimate;
    }
    bool read(std::istream& /*in*/) override {
        return false;
    }
    bool write(std::ostream& /*out*/) const override {
        return false;
    }
};
class EdgeProjection : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit EdgeProjection(Eigen::Vector3d pose) : pose3d_(std::move(pose)) {}
    void computeError() override {
        const VertexPose* v = static_cast<VertexPose*>(_vertices[0]);
        Sophus::SE3d T      = v->estimate();
        _error              = _measurement - T * pose3d_;
    }
    void linearizeOplus() override {
        const VertexPose* v                = static_cast<VertexPose*>(_vertices[0]);
        Sophus::SE3d T                     = v->estimate();
        Eigen::Vector3d trans              = T * pose3d_;
        _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
        _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(trans);
    }
    bool read(std::istream& /*in*/) override {
        return false;
    }
    bool write(std::ostream& /*out*/) const override {
        return false;
    }

private:
    Eigen::Vector3d pose3d_;
};

void bundleAdjustmentG2o(
    const std::vector<cv::Point3d>& points_3d1, const std::vector<cv::Point3d>& points_3d2, Sophus::SE3d& pose);
void svd_pose_estimation(
    const std::vector<cv::Point3d>& points_3d1, const std::vector<cv::Point3d>& points_3d2, cv::Mat& R, cv::Mat t);
int main() {

    std::string first_file        = "/workspace/slam-exercise/src/ch7/1.png";
    std::string second_file       = "/workspace/slam-exercise/src/ch7/2.png";
    std::string first_depth_file  = "/workspace/slam-exercise/src/ch7/1_depth.png";
    std::string second_depth_file = "/workspace/slam-exercise/src/ch7/2_depth.png";
    cv::Mat img_1                 = cv::imread(first_file, cv::IMREAD_COLOR);
    cv::Mat img_2                 = cv::imread(second_file, cv::IMREAD_COLOR);
    cv::Mat img_depth_1           = cv::imread(first_depth_file, cv::IMREAD_UNCHANGED);
    cv::Mat img_depth_2           = cv::imread(second_depth_file, cv::IMREAD_UNCHANGED);
    std::vector<cv::KeyPoint> keypoints_1;
    std::vector<cv::KeyPoint> keypoints_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    fmt::print("一共找到了{}组匹配点\n", matches.size());

    std::vector<cv::Point3d> points_3d1;
    std::vector<cv::Point3d> points_3d2;

    for (auto m : matches) {
        ushort d1 =
            img_depth_1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        ushort d2 =
            img_depth_2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
        if (d1 == 0) { // bad depth
            continue;
        }
        if (d2 == 0) { // bad depth
            continue;
        }
        float dd1      = d1 / 5000.0;
        float dd2      = d2 / 5000.0;
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, kIntrinsicMatrix);
        cv::Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, kIntrinsicMatrix);
        points_3d1.emplace_back(p1.x * dd1, p1.y * dd1, dd1);
        points_3d2.emplace_back(p2.x * dd2, p2.y * dd2, dd2);
    }
    fmt::print("3d-3d pair: {} \n", points_3d1.size());
    cv::Mat R;
    cv::Mat t;
    svd_pose_estimation(points_3d1, points_3d2, R, t);
    Sophus::SE3d pose;
    bundleAdjustmentG2o(points_3d1, points_3d2, pose);
    return 0;
}
void svd_pose_estimation(
    const std::vector<cv::Point3d>& points_3d1, const std::vector<cv::Point3d>& points_3d2, cv::Mat& R, cv::Mat t) {
    cv::Point3d centroid1;
    cv::Point3d centroid2;
    const int match_size = points_3d1.size();
    for (int i = 0; i < match_size; ++i) {
        centroid1 += points_3d1[i];
        centroid2 += points_3d2[i];
    }
    centroid1 = cv::Point3d(cv::Vec3d(centroid1) / match_size);
    centroid2 = cv::Point3d(cv::Vec3d(centroid2) / match_size);


    std::vector<cv::Point3d> q1(match_size);
    std::vector<cv::Point3d> q2(match_size);
    for (int i = 0; i < match_size; ++i) {
        q1[i] = points_3d1[i] - centroid1;
        q2[i] = points_3d2[i] - centroid2;
    }
    Eigen::Matrix3d W;
    for (int i = 0; i < match_size; ++i) {
        W += Eigen::Vector3d(points_3d1[i].x, points_3d1[i].y, points_3d1[i].z)
           * Eigen::Vector3d(points_3d2[i].x, points_3d2[i].y, points_3d2[i].z).transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d u_matrix = svd.matrixU();
    Eigen::Matrix3d v_matrix = svd.matrixV();
    Eigen::Matrix3d r_eigen  = u_matrix * v_matrix.transpose();
    if (r_eigen.determinant() < 0) {
        r_eigen = -r_eigen;
    }
    fmt::print("rotation matrix is: {}\n", r_eigen);

    Eigen::Vector3d t_eigen = Eigen::Vector3d(centroid1.x, centroid1.y, centroid1.z)
                            - r_eigen * Eigen::Vector3d(centroid2.x, centroid2.y, centroid2.z);
    fmt::print("move vector is: {}\n", t_eigen);
}

void bundleAdjustmentG2o(
    const std::vector<cv::Point3d>& points_3d1, const std::vector<cv::Point3d>& points_3d2, Sophus::SE3d& pose) {
    using BlockSolverType  = g2o::BlockSolverX;
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;
    auto* solver           = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);


    auto* vertex_pose = new VertexPose();
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex_pose);

    for (size_t i = 0; i < points_3d1.size(); ++i) {
        auto point_3d1               = points_3d1[i];
        auto point_3d2               = points_3d2[i];
        Eigen::Vector3d point1_eigen = Eigen::Vector3d(point_3d1.x, point_3d1.y, point_3d1.z);
        Eigen::Vector3d point2_eigen = Eigen::Vector3d(point_3d2.x, point_3d2.y, point_3d2.z);
        auto* edge                   = new EdgeProjection(point2_eigen);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(point1_eigen);
        edge->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(edge);
    }

    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    fmt::print("hello world");
    fmt::print("rotation matrix is: {}\n", vertex_pose->estimate().rotationMatrix());
    fmt::print("translation matrix is: {}\n", vertex_pose->estimate().translation());
}