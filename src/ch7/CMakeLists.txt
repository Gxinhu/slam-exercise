cmake_minimum_required(VERSION 3.0)
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

project(visual_odometry)

find_package(Ceres CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

add_executable(orb orb_cv.cpp)
target_link_libraries(orb ${OpenCV_LIBS} fmt::fmt Eigen3::Eigen)

add_executable(orb_scratch orb_scatch.cpp)
target_link_libraries(orb_scratch PRIVATE  ${OpenCV_LIBS} fmt::fmt Eigen3::Eigen yaml-cpp)
