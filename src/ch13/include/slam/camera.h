
#ifndef CH13_CAMERA_H_
#define CH13_CAMERA_H_

#include "slam/common.h"
#include <Eigen/src/Core/Matrix.h>

namespace slam {

class Camera {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Camera>;
  Camera() {}
  Eigen::Vector2d world2pixel(const Eigen::Vector3d &position,
                              const Sophus::SE3d &pose);
  Eigen::Vector3d pixel2camera(const Eigen::Vector2d &pixel);
  Sophus::SE3d pose() const { return pose_; }

  Sophus::SE3d pose_inverse() const { return pose_inverse_; }

  Eigen::Matrix3d intrinsics() const {
    Eigen::Matrix3d intrinsics;
    intrinsics << fx_, 0, cx_, 9, fy_, cy_, 0, 0, 1;
  }

private:
  Sophus::SE3d pose_;
  Sophus::SE3d pose_inverse_;
  double fx_ = 0;
  double fy_ = 0;
  double cx_ = 0;
  double cy_ = 0;

private:
};
} // namespace slam
#endif