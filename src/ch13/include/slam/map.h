#ifndef CH13_MAP_H_
#define CH13_MAP_H_
#include "slam/common.h"
#include "slam/frame.h"
#include "slam/mappoint.h"
#include <memory>
#include <mutex>
#include <unordered_map>

namespace slam {
struct MapPoint;
struct Frame;
struct Feature;
class Map {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Map>;
  using LandmarkType = std::unordered_map<unsigned long, MapPoint::Ptr>;
  using KeyframeType = std::unordered_map<unsigned long, Frame::Ptr>;

  Map() {}

  void CleanMap();
  void InsertKeyFrame(Frame::Ptr frame);
  void InsertMapPoint(MapPoint::Ptr map_point);

  LandmarkType landmarks() {
    std::unique_lock<std::mutex> lock(data_mutex_);
    return landmarks_;
  }
  void set_landmarks(const LandmarkType &landmarks) {
    std::unique_lock<std::mutex> lock(data_mutex_);
    landmarks_ = landmarks;
  }

  KeyframeType keyframes() {
    std::unique_lock<std::mutex> lock(data_mutex_);
    return keyframes_;
  }
  void set_keyframes(const KeyframeType &keyframes) {
    std::unique_lock<std::mutex> lock(data_mutex_);
    keyframes_ = keyframes;
  }

private:
  void RemoveOldKeyframe();

  std::mutex data_mutex_;
  LandmarkType landmarks_;
  LandmarkType activate_landmarks_;
  KeyframeType keyframes_;
  KeyframeType activate_keyframes_;
  Frame::Ptr current_frame_ = nullptr;

  const int num_active_keyframes_ = 7;
};

} // namespace slam

#endif