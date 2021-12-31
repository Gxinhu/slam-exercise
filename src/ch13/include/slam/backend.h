#ifndef CH13_BACKEND_H_
#define CH13_BACKEND_H_

#include "slam/common.h"
#include <Eigen/src/Core/Matrix.h>

namespace slam {

class BackEnd {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<BackEnd>;
  BackEnd() {}
  int UpdateMap();

private:
};
} // namespace slam
#endif