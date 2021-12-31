#ifndef CH13_MAPPOINT_H_
#define CH13_MAPPOINT_H_
#include "slam/common.h"
namespace slam {

struct Feature;

struct MapPoint {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<MapPoint>;
  MapPoint() {}
  MapPoint(long id, Eigen::Vector3d position);
  void AddObservation(std::shared_ptr<Feature> feature) {
    std::unique_lock<std::mutex> lock(data_mutex_);
    observations_.push_back(feature);
    observed_times_++;
  }
  void RemoveObservation(std::shared_ptr<Feature> feature);
  auto observations() {
    std::unique_lock<std::mutex> lock(data_mutex_);
    return observations_;
  }
  Eigen::Vector3d position() {
    std::unique_lock<std::mutex> lock(data_mutex_);
    return position_;
  }
  void set_position(const Eigen::Vector3d &position) {
    std::unique_lock<std::mutex> lock(data_mutex_);
    position_ = position;
  }
  static MapPoint::Ptr CreateNewMapPoint();

private:
  unsigned long id_ = 0;
  bool is_outlier_ = false;
  int observed_times_ = 0;
  Eigen::Vector3d position_ = Eigen::Vector3d::Zero();
  std::list<std::weak_ptr<Feature>> observations_;
  std::mutex data_mutex_;
};
} // namespace slam
#endif // CH13_MAPPOINT_H_