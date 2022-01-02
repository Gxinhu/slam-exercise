#ifndef CH13_G2OTYPES_H_
#define CH13_G2OTYPES_H_
#include "slam/common.h"
#include <Eigen/src/Core/Matrix.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <sophus/se3.hpp>
namespace slam {

/// vertex and edges used in g2o ba
/// 位姿顶点
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override { _estimate = Sophus::SE3d(); }

  /// left multiplication on SE3
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4],
        update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }

  virtual bool read(std::istream &in) override { return true; }

  virtual bool write(std::ostream &out) const override { return true; }
};

/// 路标顶点
class VertexXYZ : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  virtual void setToOriginImpl() override {
    _estimate = Eigen::Vector3d::Zero();
  }

  virtual void oplusImpl(const double *update) override {
    _estimate[0] += update[0];
    _estimate[1] += update[1];
    _estimate[2] += update[2];
  }

  virtual bool read(std::istream &in) override { return true; }

  virtual bool write(std::ostream &out) const override { return true; }
};

/// 仅估计位姿的一元边
class EdgeProjectionPoseOnly
    : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjectionPoseOnly(const Eigen::Vector3d &pos,
                         const Eigen::Matrix3d &intrinsics)
      : pose3d_(pos), intrinsics_(intrinsics) {}

  virtual void computeError() override {
    const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_pixel = intrinsics_ * (T * pose3d_);
    pos_pixel /= pos_pixel[2];
    _error = _measurement - pos_pixel.head<2>();
  }

  virtual void linearizeOplus() override {
    const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_cam = T * pose3d_;
    double fx = intrinsics_(0, 0);
    double fy = intrinsics_(1, 1);
    double x = pos_cam[0];
    double y = pos_cam[1];
    double z = pos_cam[2];
    double z_inverse = 1.0 / (z + 1e-18);
    double z_inverse_square = z_inverse * z_inverse;
    _jacobianOplusXi << -fx * z_inverse, 0, fx * x * z_inverse_square,
        fx * x * y * z_inverse_square, -fx - fx * x * x * z_inverse_square,
        fx * y * z_inverse, 0, -fy * z_inverse, fy * y * z_inverse_square,
        fy + fy * y * y * z_inverse_square, -fy * x * y * z_inverse_square,
        -fy * x * z_inverse;
  }

  virtual bool read(std::istream &in) override { return true; }

  virtual bool write(std::ostream &out) const override { return true; }

private:
  Eigen::Vector3d pose3d_;
  Eigen::Matrix3d intrinsics_;
};

/// 带有地图和位姿的二元边
class EdgeProjection
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexPose, VertexXYZ> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  /// 构造时传入相机内外参
  EdgeProjection(const Eigen::Matrix3d &intrinsics, const Sophus::SE3d &cam_ext)
      : intrinsics_(intrinsics) {
    camera_pose_ = cam_ext;
  }

  virtual void computeError() override {
    const VertexPose *v0 = static_cast<VertexPose *>(_vertices[0]);
    const VertexXYZ *v1 = static_cast<VertexXYZ *>(_vertices[1]);
    Sophus::SE3d T = v0->estimate();
    Eigen::Vector3d pos_pixel =
        intrinsics_ * (camera_pose_ * (T * v1->estimate()));
    pos_pixel /= pos_pixel[2];
    _error = _measurement - pos_pixel.head<2>();
  }

  virtual void linearizeOplus() override {
    const VertexPose *v0 = static_cast<VertexPose *>(_vertices[0]);
    const VertexXYZ *v1 = static_cast<VertexXYZ *>(_vertices[1]);
    Sophus::SE3d T = v0->estimate();
    Eigen::Vector3d world_point = v1->estimate();
    Eigen::Vector3d pos_cam = camera_pose_ * T * world_point;
    double fx = intrinsics_(0, 0);
    double fy = intrinsics_(1, 1);
    double x = pos_cam[0];
    double y = pos_cam[1];
    double z = pos_cam[2];
    double z_inverse = 1.0 / (z + 1e-18);
    double z_inverse_square = z_inverse * z_inverse;
    _jacobianOplusXi << -fx * z_inverse, 0, fx * x * z_inverse_square,
        fx * x * y * z_inverse_square, -fx - fx * x * x * z_inverse_square,
        fx * y * z_inverse, 0, -fy * z_inverse, fy * y * z_inverse_square,
        fy + fy * y * y * z_inverse_square, -fy * x * y * z_inverse_square,
        -fy * x * z_inverse;

    _jacobianOplusXj = _jacobianOplusXi.block<2, 3>(0, 0) *
                       camera_pose_.rotationMatrix() * T.rotationMatrix();
  }

  virtual bool read(std::istream &in) override { return true; }

  virtual bool write(std::ostream &out) const override { return true; }

private:
  Eigen::Matrix3d intrinsics_;
  Sophus::SE3d camera_pose_;
};

} // namespace slam
#endif // CH13_FRONTEND_H_
