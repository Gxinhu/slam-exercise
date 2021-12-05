cmake_minimum_required(VERSION 3.0)

project(lie_theory)
add_executable(use_sophus use_sophus.cpp)
target_link_libraries(use_sophus Eigen3::Eigen Sophus::Sophus) 
