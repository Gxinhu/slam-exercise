#include "slam/frontend.h"
#include "slam/common.h"
#include "slam/feature.h"
#include <cmath>
#include <iostream>
namespace slam {

bool Frontend::AddFrame(Frame::Ptr frame) {
  current_frame_ = frame;
  switch (status_) {
  case FrontendStatus::INITING:
    StereoInit();
    break;
  case FrontendStatus::TRACKING_GOOD:
  case FrontendStatus::TRACKING_BAD:
    Track();
    break;
  case FrontendStatus::LOST:
    Reset();
    break;
  }
  last_frame_ = current_frame_;
  return true;
}
bool Frontend::Track() {
  if (last_frame_) {
    current_frame_->set_pose(relative_motion_ * last_frame_->pose());
  }
  int num_track_last = TrackLastFrame();
  tracking_inliers_ = EstimateCurrentPose();
  if (tracking_inliers_ > num_features_tracking_) {
    status_ = FrontendStatus::TRACKING_GOOD;
  } else {
    status_ = FrontendStatus::TRACKING_BAD;
  }
  InsertKeyframe();
  relative_motion_ = current_frame_->pose() * last_frame_->pose().inverse();
  if (viewer_) {
    viewer_->AddCurrentFrame(current_frame_);
  }
  return true;
}
int Frontend::TrackLastFrame() {
  std::vector<cv::Point2f> last_keypoints;
  std::vector<cv::Point2f> current_keypoints;
  for (auto &keypoint : last_frame_->features_left()) {
    if (keypoint->map_point().lock()) {
      auto map_point = keypoint->map_point().lock();
      auto pixel = camera_left_->world2pixel(map_point->position(),
                                             current_frame_->pose());
      last_keypoints.push_back(keypoint->position().pt);
      current_keypoints.push_back(keypoint->position().pt);
    } else {
      last_keypoints.push_back(keypoint->position().pt);
      current_keypoints.push_back(keypoint->position().pt);
    }
  }
  std::vector<uchar> status;
  cv::Mat error;
  cv::calcOpticalFlowPyrLK(
      last_frame_->left_img(), current_frame_->left_img(), last_keypoints,
      current_keypoints, status, error, cv::Size(21, 21), 3,
      cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                       0.01),
      cv::OPTFLOW_USE_INITIAL_FLOW);
  int num_good_points = 0;
  for (size_t i = 0; i < status.size(); ++i) {
    if (status[i]) {
      cv::KeyPoint keypoint = cv::KeyPoint(current_keypoints[i], 7);
      Feature::Ptr feature(new Feature(current_frame_, keypoint));
      feature->set_map_point(last_frame_->features_left()[i]->map_point());
      current_frame_->AddFeaturesLeft(feature);
      num_good_points++;
    }
  }
  logger->info("num_good_pointsFind {} in the last image.", num_good_points);
  return num_good_points;
}
bool Frontend::InsertKeyframe() {
  if (tracking_inliers_ >= num_features_needed_keyframe_) {
    return false;
  }
  current_frame_->set_keyframe();
  map_->InsertKeyFrame(current_frame_);
  logger->info("Set frame {} as keyframe {} \n", current_frame_->id(),
               current_frame_->keyframe_id());
  SetObservationsForKeyFrame();
  DetectFeatures();
  FindFeatureInRights();
  backend_->UpdateMap();
  if (viewer_) {
    viewer_->UpdateMap();
  }
  return true;
}

void Frontend::SetObservationsForKeyFrame() {
  for (auto &feature : current_frame_->features_left()) {
    auto map_point = feature->map_point().lock();
    if (map_point) {
      map_point->AddObservation(feature);
    }
  }
}

int Frontend::TriangulateNewPoints() {
  std::vector<Sophus::SE3d> poses{camera_left_->pose(), camera_right_->pose()};

}
} // namespace slam