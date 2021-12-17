#include "include/pose_estimation_2d.h"
#include <Eigen/src/Core/Array.h>
#include <Eigen/src/Core/Matrix.h>
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
#include <ostream>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sophus/se3.hpp>
using VecVector3d = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
using VecVector2d = std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>;
void bundleAdjustmentGaussNewton(
    const VecVector3d& points_3d, const VecVector2d& points_2d, const cv::Mat& intrinsic_matrix, Sophus::SE3d& pose);
// 顶点: 需要优化的参数，在 PnP中也就是位姿
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    void oplusImpl(const double* update) override {
        Eigen::Matrix<double, 6, 1> update_vector;
        update_vector << update[0], update[1], update[2], update[3], update[4],
            update[5];
        _estimate = Sophus::SE3d::exp(update_vector) * _estimate;
    }
    bool read(std::istream& /*in*/) override {
        return false;
    }
    bool write(std::ostream& /*out*/) const override {
        return false;
    }
};
class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjection(Eigen::Vector3d pose, Eigen::Matrix3d K) : pose3d_(std::move(pose)), K_(std::move(K)) {}
    void computeError() override {
        const VertexPose* v        = static_cast<VertexPose*>(_vertices[0]);
        Sophus::SE3d T             = v->estimate();
        Eigen::Vector3d pose_pixel = K_ * (T * pose3d_);
        pose_pixel /= pose_pixel[2];
        _error = _measurement - pose_pixel.head<2>();
    }
    void linearizeOplus() override {
        const VertexPose* v          = static_cast<VertexPose*>(_vertices[0]);
        Sophus::SE3d T               = v->estimate();
        Eigen::Vector3d camera_point = T * pose3d_;
        double fx                    = K_(0, 0);
        double fy                    = K_(1, 1);
        double cx                    = K_(0, 2);
        double cy                    = K_(1, 2);
        double inv_z                 = 1.0 / camera_point[2];
        double inv_z_square          = inv_z * inv_z;
        _jacobianOplusXi << -fx * inv_z, 0, fx * camera_point[0] * inv_z_square,
            fx * camera_point[0] * camera_point[1] * inv_z_square, -fx - fx * pow(camera_point[0], 2) * inv_z_square,
            fx * camera_point[1] * inv_z, 0, -fy * inv_z, fy * camera_point[1] * inv_z,
            fy + fy * pow(camera_point[1], 2) * inv_z_square, -fy * camera_point[0] * camera_point[1] * inv_z_square,
            -fy * camera_point[0] * inv_z;
    }
    bool read(std::istream& /*in*/) override {
        return false;
    }
    bool write(std::ostream& /*out*/) const override {
        return false;
    }

private:
    Eigen::Vector3d pose3d_;
    Eigen::Matrix3d K_;
};

void bundleAdjustmentG2o(
    const VecVector2d& points_2d, const VecVector3d& points_3d, const cv::Mat& K, Sophus::SE3d& pose);
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

    std::vector<cv::Point3d> points_3d;
    std::vector<cv::Point2d> points_2d;

    for (auto m : matches) {
        ushort d =
            img_depth_1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0) { // bad depth
            continue;
        }
        float dd       = d / 5000.0;
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, kIntrinsicMatrix);
        points_3d.emplace_back(p1.x * dd, p1.y * dd, dd);
        points_2d.push_back(keypoints_2[m.trainIdx].pt);
    }
    fmt::print("3d-2d pair: {} \n", points_3d.size());
    auto t1 = std::chrono::steady_clock::now();
    cv::Mat rotation_vector;
    cv::Mat move_vector;
    cv::solvePnP(points_3d, points_2d, kIntrinsicMatrix, cv::Mat(), rotation_vector, move_vector);
    cv::Mat rotation_matrix;
    cv::Rodrigues(rotation_vector, rotation_matrix);
    auto t2        = std::chrono::steady_clock::now();
    auto used_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    fmt::print("solve PnP need time :{}\n", used_time.count());
    fmt::print("Rotation Matrix is:{}", rotation_matrix);
    fmt::print("Move Vector is:{}", move_vector);
    VecVector2d points_2d_eigen;
    VecVector3d points_3d_eigen;
    for (int i = 0; i < points_3d.size(); ++i) {
        points_3d_eigen.push_back(Eigen::Vector3d(points_3d[i].x, points_3d[i].y, points_3d[i].z));
        points_2d_eigen.push_back(Eigen::Vector2d(points_2d[i].x, points_2d[i].y));
    }

    Sophus::SE3d pose;
    bundleAdjustmentGaussNewton(points_3d_eigen, points_2d_eigen, kIntrinsicMatrix, pose);
    bundleAdjustmentG2o(points_2d_eigen, points_3d_eigen, kIntrinsicMatrix, pose);
    return 0;
}
void bundleAdjustmentGaussNewton(
    const VecVector3d& points_3d, const VecVector2d& points_2d, const cv::Mat& intrinsic_matrix, Sophus::SE3d& pose) {
    using Vector6d      = Eigen::Matrix<double, 6, 1>;
    const int iteration = 10;
    double cost         = 0;
    double last_cost    = 0;
    double fx           = intrinsic_matrix.at<double>(0, 0);
    double fy           = intrinsic_matrix.at<double>(1, 1);
    double cx           = intrinsic_matrix.at<double>(0, 2);
    double cy           = intrinsic_matrix.at<double>(1, 2);

    for (int iter = 0; iter < iteration; ++iter) {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b                    = Vector6d::Zero();
        cost                          = 0;

        for (int i = 0; i < points_3d.size(); ++i) {
            Eigen::Vector3d camera_point = pose * points_3d[i];
            double inv_z                 = 1.0 / camera_point[2];
            double inv_z_square          = inv_z * inv_z;
            Eigen::Vector2d pixel(fx * camera_point[0] * inv_z + cx, fy * camera_point[1] * inv_z + cy);
            Eigen::Vector2d error = points_2d[i] - pixel;
            cost += error.squaredNorm();
            Eigen::Matrix<double, 2, 6> jacobian_matrix;
            jacobian_matrix << -fx * inv_z, 0, fx * camera_point[0] * inv_z_square,
                fx * camera_point[0] * camera_point[1] * inv_z_square,
                -fx - fx * pow(camera_point[0], 2) * inv_z_square, fx * camera_point[1] * inv_z, 0, -fy * inv_z,
                fy * camera_point[1] * inv_z, fy + fy * pow(camera_point[1], 2) * inv_z_square,
                -fy * camera_point[0] * camera_point[1] * inv_z_square, -fy * camera_point[0] * inv_z;
            H += jacobian_matrix.transpose() * jacobian_matrix;
            b += -jacobian_matrix.transpose() * error;
        }
        Vector6d dx = H.ldlt().solve(b);
        if (std::isnan(dx[0])) {
            std::cerr << "result is nan!" << std::endl;
            break;
        }
        if (iter > 0 && cost >= last_cost) {
            fmt::print("cost:{},last cost:{}\n", cost, last_cost);
            break;
        }
        pose      = Sophus::SE3d::exp(dx) * pose;
        last_cost = cost;
        fmt::print("iteration {}, cost={}\n", iter, cost);
        const double thread = 1e-6;
        if (dx.norm() < thread) {
            break;
        }
    }
    fmt::print("pose :{}\n", pose.matrix());
}
void bundleAdjustmentG2o(
    const VecVector2d& points_2d, const VecVector3d& points_3d, const cv::Mat& K, Sophus::SE3d& pose) {
    using BlockSolverType  = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
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
    Eigen::Matrix3d k_eigen;
    k_eigen << K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2), K.at<double>(1, 0), K.at<double>(1, 1),
        K.at<double>(1, 2), K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    int index = 1;
    for (size_t i = 0; i < points_2d.size(); ++i) {
        auto point_2d = points_2d[i];
        auto point_3d = points_3d[i];
        auto* edge    = new EdgeProjection(point_3d, k_eigen);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(point_2d);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }

    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    fmt::print("pose estimation by g2o ={}\n", vertex_pose->estimate().matrix());
}