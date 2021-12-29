cmake_minimum_required(VERSION 3.0)
project(loop_closure)
set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.so" )

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

find_package(octomap REQUIRED)
add_definitions(${PCL_DEFINITIONS})
add_executable(dense_mono dense_mono.cpp)
target_link_libraries(dense_mono PRIVATE ${OpenCV_LIBS} fmt::fmt Eigen3::Eigen Ceres::ceres ${DBoW3_LIBS} Sophus::Sophus argparse::argparse)

add_executable(pointcloud pointcloud_mapping.cpp)
target_link_libraries(pointcloud PRIVATE ${OpenCV_LIBS} fmt::fmt Eigen3::Eigen Ceres::ceres ${DBoW3_LIBS} Sophus::Sophus argparse::argparse ${PCL_LIBRARIES} octomap)
