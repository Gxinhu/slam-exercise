cmake_minimum_required(VERSION 3.0)

project(visual_odometry)

find_package(Ceres CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)

add_executable(orb orb_cv.cpp)
target_link_libraries(orb ${OpenCV_LIBS} fmt::fmt Eigen3::Eigen)

add_executable(orb_scratch orb_scatch.cpp)
target_link_libraries(orb_scratch ${OpenCV_LIBS} fmt::fmt Eigen3::Eigen)
