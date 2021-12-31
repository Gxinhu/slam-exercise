
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
  Sophus::SE3d pose() const { return pose_; }

  Sophus::SE3d pose_inverse() const { return pose_inverse_; }

private:
  Sophus::SE3d pose_;
  Sophus::SE3d pose_inverse_;

private:
};
} // namespace slam
#endif