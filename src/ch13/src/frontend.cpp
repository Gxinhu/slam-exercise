#include "slam/frontend.h"
#include "slam/algorithm.h"
#include "slam/common.h"
#include "slam/feature.h"
#include "slam/g2o_types.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/stuff/misc.h>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
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
  Sophus::SE3d current_pose_camera2world = current_frame_->pose().inverse();
  int current_triangulated_points = 0;
  for (size_t i = 0; i < current_frame_->features_left().size(); ++i) {
    if (current_frame_->features_left()[i]->map_point().expired() &&
        current_frame_->features_right()[i] != nullptr) {
      auto current_left_pixel =
          current_frame_->features_left()[i]->position().pt;
      auto current_right_pixel =
          current_frame_->features_right()[i]->position().pt;
      std::vector<Eigen::Vector3d> points{
          camera_left_->pixel2camera(
              Eigen::Vector2d(current_left_pixel.x, current_left_pixel.y)),
          camera_right_->pixel2camera(
              Eigen::Vector2d(current_right_pixel.x, current_right_pixel.y))};
      Eigen::Vector3d point_world = Eigen::Vector3d::Zero();
      // TODO: 这里的 三角化的 return true 和 false 有点让我摸不着头脑
      if (triangulation(poses, points, point_world) && point_world[2] > 0) {
        auto new_map_point = MapPoint::CreateNewMapPoint();
        point_world = current_pose_camera2world * point_world;
        new_map_point->set_position(point_world);
        new_map_point->AddObservation(current_frame_->features_left()[i]);
        new_map_point->AddObservation(current_frame_->features_right()[i]);
        current_frame_->features_left()[i]->set_map_point(new_map_point);
        current_frame_->features_right()[i]->set_map_point(new_map_point);
        map_->InsertMapPoint(new_map_point);
        current_triangulated_points++;
      }
    }
  }
  logger->info("new landmarks: {}\n", current_triangulated_points);
  return current_triangulated_points;
}

int Frontend::EstimateCurrentPose() {
  // pose is 6, landmark is 3
  using BlockSolverType = g2o::BlockSolver_6_3;
  using LinearSolverType = g2o::LinearSolver<BlockSolverType::PoseMatrixType>;
  auto solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);

  VertexPose *vertex_pose = new VertexPose();
  vertex_pose->setId(0);
  vertex_pose->setEstimate(current_frame_->pose());
  optimizer.addVertex(vertex_pose);

  Eigen::Matrix3d intrinsics = camera_left_->intrinsics();

  int index = 1;
  std::vector<EdgeProjectionPoseOnly *> edges;
  std::vector<Feature::Ptr> features;
  for (size_t i = 0; i < current_frame_->features_left().size(); ++i) {
    auto map_point = current_frame_->features_left()[i]->map_point().lock();
    if (map_point) {
      features.push_back(current_frame_->features_left()[i]);
      EdgeProjectionPoseOnly *edge =
          new EdgeProjectionPoseOnly(map_point->position(), intrinsics);
      edge->setId(index);
      edge->setVertex(0, vertex_pose);
      cv::Point2d point = current_frame_->features_left()[i]->position().pt;
      edge->setMeasurement(Eigen::Vector2d(point.x, point.y));
      edge->setInformation(Eigen::Matrix2d::Identity());
      edge->setRobustKernel(new g2o::RobustKernelHuber);
      edges.push_back(edge);
      optimizer.addEdge(edge);
      index++;
    }
  }

  // estimate the Pose the determine the outliers
  const double chi2_th = 5.991;
  int cnt_outlier = 0;
  for (int iteration = 0; iteration < 4; ++iteration) {
    vertex_pose->setEstimate(current_frame_->pose());
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    cnt_outlier = 0;

    // count the outliers
    for (size_t i = 0; i < edges.size(); ++i) {
      auto e = edges[i];
      if (features[i]->is_outlier()) {
        e->computeError();
      }
      if (e->chi2() > chi2_th) {
        features[i]->set_is_outlier(true);
        e->setLevel(1);
        cnt_outlier++;
      } else {
        features[i]->set_is_outlier(false);
        e->setLevel(0);
      };

      if (iteration == 2) {
        e->setRobustKernel(nullptr);
      }
    }
  }

  logger->info("Outlier/Inlier in pose estimating: {} / {}", cnt_outlier,
               features.size() - cnt_outlier);
  // Set pose and outlier
  current_frame_->set_pose(vertex_pose->estimate());

  logger->info("Current Pose = \n {}", current_frame_->pose().matrix());

  for (auto &feat : features) {
    if (feat->is_outlier()) {
      feat->map_point().reset();
      feat->set_is_outlier(false); // maybe we can still use it in future
    }
  }
  return features.size() - cnt_outlier;
}

int Frontend::DetectFeatures() {
  cv::Mat mask(current_frame_->left_img().size(), CV_8UC1, 255);
  for (auto &feat : current_frame_->features_left()) {
    cv::rectangle(mask, feat->position().pt - cv::Point2f(10, 10),
                  feat->position().pt + cv::Point2f(10, 10), 0, cv::FILLED);
  }

  std::vector<cv::KeyPoint> keypoints;
  gftt_detector_->detect(current_frame_->left_img(), keypoints, mask);
  int cnt_detected = 0;
  for (auto &kp : keypoints) {
    current_frame_->features_left().push_back(
        Feature::Ptr(new Feature(current_frame_, kp)));
    cnt_detected++;
  }

  logger->info("Detect {} new features\n", cnt_detected);
  return cnt_detected;
}

int Frontend::FindFeatureInRights() {
  // use LK flow to estimate points in the right image
  std::vector<cv::Point2f> kps_left, kps_right;
  for (auto &kp : current_frame_->features_left()) {
    kps_left.push_back(kp->position().pt);
    auto mp = kp->map_point().lock();
    if (mp) {
      // use projected points as initial guess
      auto px =
          camera_right_->world2pixel(mp->position(), current_frame_->pose());
      kps_right.push_back(cv::Point2f(px[0], px[1]));
    } else {
      // use same pixel in left iamge
      kps_right.push_back(kp->position().pt);
    }
  }

  std::vector<uchar> status;
  cv::Mat error;
  cv::calcOpticalFlowPyrLK(
      current_frame_->left_img(), current_frame_->right_img(), kps_left,
      kps_right, status, error, cv::Size(11, 11), 3,
      cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                       0.01),
      cv::OPTFLOW_USE_INITIAL_FLOW);

  int num_good_pts = 0;
  for (size_t i = 0; i < status.size(); ++i) {
    if (status[i]) {
      cv::KeyPoint kp(kps_right[i], 7);
      Feature::Ptr feat(new Feature(current_frame_, kp));
      feat->set_is_on_left_image(false);
      current_frame_->features_right().push_back(feat);
      num_good_pts++;
    } else {
      current_frame_->features_right().push_back(nullptr);
    }
  }
  logger->info("Find {} in the right image.\n", num_good_pts);
  return num_good_pts;
}

bool Frontend::BuildInitMap() {
  std::vector<Sophus::SE3d> poses{camera_left_->pose(), camera_right_->pose()};
  size_t cnt_init_landmarks = 0;
  for (size_t i = 0; i < current_frame_->features_left().size(); ++i) {
    if (current_frame_->features_right()[i] == nullptr)
      continue;
    // create map point from triangulation
    std::vector<Eigen::Vector3d> points{
        camera_left_->pixel2camera(Eigen::Vector2d(
            current_frame_->features_left()[i]->position().pt.x,
            current_frame_->features_left()[i]->position().pt.y)),
        camera_right_->pixel2camera(Eigen::Vector2d(
            current_frame_->features_right()[i]->position().pt.x,
            current_frame_->features_right()[i]->position().pt.y))};
    Eigen::Vector3d pworld = Eigen::Vector3d::Zero();

    if (triangulation(poses, points, pworld) && pworld[2] > 0) {
      auto new_map_point = MapPoint::CreateNewMapPoint();
      new_map_point->set_position(pworld);
      new_map_point->AddObservation(current_frame_->features_left()[i]);
      new_map_point->AddObservation(current_frame_->features_right()[i]);
      current_frame_->features_left()[i]->map_point() = new_map_point;
      current_frame_->features_right()[i]->map_point() = new_map_point;
      cnt_init_landmarks++;
      map_->InsertMapPoint(new_map_point);
    }
  }
  current_frame_->set_keyframe();
  map_->InsertKeyFrame(current_frame_);
  backend_->UpdateMap();

  logger->info("Initial map created with {} map points.\n", cnt_init_landmarks);
  return true;
}

bool Frontend::Reset() {
  logger->info("Reset is not implementsed. \n");
  return true;
}

} // namespace slam