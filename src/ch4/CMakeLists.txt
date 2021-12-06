cmake_minimum_required(VERSION 3.0)

project(lie_theory)
add_executable(use_sophus use_sophus.cpp)
target_link_libraries(use_sophus Eigen3::Eigen Sophus::Sophus) 

add_executable(plot_trajectory plot_trajectory_error.cpp)
target_link_libraries(plot_trajectory Eigen3::Eigen Sophus::Sophus pangolin) 
target_include_directories(plot_trajectory PRIVATE ${Pangolin_INCLUDE_DIRS})