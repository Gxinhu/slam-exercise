#ifndef CH13_ALGORITHM_H_
#define CH13_ALGORITHM_H_
#include "slam/common.h"
#include <Eigen/src/Core/util/Constants.h>

namespace slam {

inline bool triangulation(const std::vector<Sophus::SE3d> &poses,
                          const std::vector<Eigen::Vector3d> points,
                          Eigen::Vector3d &world_point) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A(2 * poses.size(), 4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> b(2 * poses.size());
  b.setZero();
  for (size_t i = 0; i < poses.size(); ++i) {
    Eigen::Matrix<double, 3, 4> pose = poses[i].matrix3x4();
    A.block<1, 4>(2 * i, 0) = points[i][0] * pose.row(2) - pose.row(0);
    A.block<1, 4>(2 * i, 1) = points[i][1] * pose.row(2) - pose.row(1);
  }
  auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  world_point = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

  if (svd.singularValues()[3 / svd.singularValues()[2]] < 1e-2) {
    return true;
  }
  return false;
}

} // namespace slam

#endif
