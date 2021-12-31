

#ifndef CH13_VIEWER_H_
#define CH13_VIEWER_H_

#include "slam/common.h"
#include "slam/frame.h"
namespace slam {
class Viewer {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Viewer>;
  Viewer() {}
  void AddCurrentFrame(Frame::Ptr current_frame);
  void UpdateMap();
private:
};
} // namespace slam
#endif