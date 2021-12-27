
cmake_minimum_required(VERSION 3.0)

project(bundle_adjust)

find_package(Ceres CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

add_library(common SHARED common.cpp)
target_link_libraries(common PRIVATE ${OpenCV_LIBS} fmt::fmt Eigen3::Eigen Ceres::ceres)

add_executable(ceres_ba snavely_reprojection_error.cpp)
target_link_libraries(ceres_ba PRIVATE  common ${OpenCV_LIBS} fmt::fmt Eigen3::Eigen Ceres::ceres)
