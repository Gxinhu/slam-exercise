#ifndef CH13_FEATURE_H_
#define CH13_FEATURE_H_

#include "slam/common.h"
#include "slam/mappoint.h"
namespace slam {
struct Frame;
struct Feature {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Feature>;
  Feature() {}
  Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &position)
      : frame_(frame), position_(position) {}

  std::weak_ptr<MapPoint> map_point() const { return map_point_; }
  void set_map_point(const std::weak_ptr<MapPoint> &map_point) {
    map_point_ = map_point;
  }

  std::weak_ptr<Frame> frame() const { return frame_; }

  cv::KeyPoint position() const { return position_; }

  bool is_outlier() const { return is_outlier_; }
  void set_is_outlier(bool is_outlier) { is_outlier_ = is_outlier; }

  bool is_on_left_image() const { return is_on_left_image_; }
  void set_is_on_left_image(bool is_on_left_image) {
    is_on_left_image_ = is_on_left_image;
  }

private:
  cv::KeyPoint position_;
  std::weak_ptr<MapPoint> map_point_;
  std::weak_ptr<Frame> frame_;
  bool is_outlier_ = false;
  bool is_on_left_image_ = true;
};
} // namespace slam
#endif