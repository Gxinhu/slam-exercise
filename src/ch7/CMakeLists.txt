cmake_minimum_required(VERSION 3.0)
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++17 ${SSE_FLAGS} -msse4")

project(visual_odometry)

find_package(Ceres CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
add_library(pose2d SHARED pose_estimation_2d.cpp)
target_link_libraries(pose2d  PUBLIC ${OpenCV_LIBS} fmt::fmt Eigen3::Eigen yaml-cpp Sophus::Sophus g2o::core)
add_executable(orb orb_cv.cpp)
target_link_libraries(orb pose2d)

add_executable(orb_scratch orb_scratch.cpp)
target_link_libraries(orb_scratch pose2d)


add_executable(pose3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose3d2d pose2d)

add_executable(pose3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose3d3d pose2d)