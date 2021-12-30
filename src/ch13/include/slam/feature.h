
#ifndef CH13_FEATURE_H_
#define CH13_FEATURE_H_

#include <Eigen/Core>
#include <cstdint>
#include <memory>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <sophus/se3.hpp>
#include <vector>

struct Frame {

public:
    Frame() = default;
    Frame(const Sophus::SE3d& pose, const cv::Mat& left, const cv::Mat& right, long id,
        double time_stamp);
    Sophus::SE3d pose() {
        std::unique_lock<std::mutex> lock(pose_mutex_);
        return pose_;
    }
    void set_pose(const Sophus::SE3d pose) {
        std::unique_lock<std::mutex> lock(pose_mutex_);
        pose_ = pose;
    }
    void set_keyframe();
    static std::shared_ptr<Frame> createFrame();
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Frame>;

    std::uint64_t id_          = 0;
    std::uint64_t keyframe_id_ = 0;
    bool is_keyframe_          = false;
    double time_stamp_;
    Sophus::SE3d pose_;
    std::mutex pose_mutex_;
    cv::Mat left_img_;
    cv::Mat right_img_;

    std::vector<std::shared_ptr<Feature>> features_left_;
    std::vector<std::shared_ptr<Feature>> features_right_;
};
#endif