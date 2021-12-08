
cmake_minimum_required(VERSION 3.0)

project(camera_vision)

add_executable(image_basic image_basic.cpp)
target_link_libraries(image_basic ${OpenCV_LIBS} fmt::fmt) 
add_executable(undistort_image undistort_image.cpp)
target_link_libraries(undistort_image ${OpenCV_LIBS} fmt::fmt) 

add_executable(stereo stereo_vision.cpp)
target_link_libraries(stereo ${OpenCV_LIBS} fmt::fmt Eigen3::Eigen pangolin) 

add_executable(rgbd rgbd_point_cloud.cpp)
target_link_libraries(rgbd ${OpenCV_LIBS} fmt::fmt Eigen3::Eigen pangolin Sophus::Sophus) 