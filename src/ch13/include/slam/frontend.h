
#ifndef CH13_FRONTEND_H_
#define CH13_FRONTEND_H_
#include "slam/backend.h"
#include "slam/camera.h"
#include "slam/common.h"
#include "slam/frame.h"
#include "slam/map.h"
#include "slam/viewer.h"
namespace slam {
class Frontend {
  enum class FrontendStatus { INITING, TRACKING_GOOD, TRACKING_BAD, LOST };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Frontend>;
  Frontend() {}
  bool AddFrame(Frame::Ptr frame);

  void set_map(Map::Ptr map) { map_ = map; }

  FrontendStatus status() const { return status_; }

  void set_backend(const std::shared_ptr<BackEnd> &backend) {
    backend_ = backend;
  }

  void set_viewer(const std::shared_ptr<Viewer> &viewer) { viewer_ = viewer; }

private:
  bool Track();
  bool Reset();
  int TrackLastFrame();
  int EstimateCurrentPose();
  bool InsertKeyframe();
  bool StereoInit();
  int FindFeatureInRights();
  bool BuildInitMap();
  int TriangulateNewPoints();
  void SetObservationsForKeyFrame();
  int DetectFeatures();

  FrontendStatus status_ = FrontendStatus::INITING;
  Frame::Ptr current_frame_;
  Frame::Ptr last_frame_;
  Map::Ptr map_ = nullptr;
  Camera::Ptr camera_left_ = nullptr;
  Camera::Ptr camera_right_ = nullptr;
  std::shared_ptr<BackEnd> backend_ = nullptr;
  std::shared_ptr<Viewer> viewer_ = nullptr;

  Sophus::SE3d relative_motion_;
  int tracking_inliers_ = 0;

  const int num_features_ = 200;
  const int num_features_init = 100;
  const int num_features_tracking_ = 50;
  const int num_features_tracking_bad_ = 20;
  const int num_features_needed_keyframe_ = 80;

  cv::Ptr<cv::GFTTDetector> gftt_detector;
};
} // namespace slam

#endif // CH13_FRONTEND_H_