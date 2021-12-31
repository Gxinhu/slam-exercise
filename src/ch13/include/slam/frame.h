#ifndef CH13_FRAME_H_
#define CH13_FRAME_H_
#include "slam/common.h"
#include "slam/feature.h"
namespace slam {

struct Frame {

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Frame>;
  Frame() = default;
  Frame(const Sophus::SE3d &pose, const cv::Mat &left, const cv::Mat &right,
        long id, double time_stamp);
  Sophus::SE3d pose() {
    std::unique_lock<std::mutex> lock(pose_mutex_);
    return pose_;
  }
  void set_pose(const Sophus::SE3d &pose) {
    std::unique_lock<std::mutex> lock(pose_mutex_);
    pose_ = pose;
  }
  void set_keyframe();
  static std::shared_ptr<Frame> createFrame();

  std::uint64_t id() const { return id_; }
  void set_id(const std::uint64_t &id) { id_ = id; }

  std::uint64_t keyframe_id() const { return keyframe_id_; }
  void set_keyframe_id(const std::uint64_t &keyframe_id) {
    keyframe_id_ = keyframe_id;
  }

  bool is_keyframe() const { return is_keyframe_; }
  void set_is_keyframe(bool is_keyframe) { is_keyframe_ = is_keyframe; }

  double time_stamp() const { return time_stamp_; }
  void set_time_stamp(double time_stamp) { time_stamp_ = time_stamp; }

  cv::Mat left_img() const { return left_img_; }
  void set_left_img(const cv::Mat &left_img) { left_img_ = left_img; }

  cv::Mat right_img() const { return right_img_; }
  void set_right_img(const cv::Mat &right_img) { right_img_ = right_img; }

  std::vector<std::shared_ptr<Feature>> features_right() const {
    return features_right_;
  }
  void AddFeaturesLeft(const Feature::Ptr &feature) {
    features_left_.push_back(feature);
  }
  std::vector<std::shared_ptr<Feature>> features_left() const {
    return features_left_;
  }

private:
  std::uint64_t id_ = 0;
  std::uint64_t keyframe_id_ = 0;
  bool is_keyframe_ = false;
  double time_stamp_;
  Sophus::SE3d pose_;
  std::mutex pose_mutex_;
  cv::Mat left_img_;
  cv::Mat right_img_;

  std::vector<std::shared_ptr<Feature>> features_left_;
  std::vector<std::shared_ptr<Feature>> features_right_;
};
} // namespace slam
#endif