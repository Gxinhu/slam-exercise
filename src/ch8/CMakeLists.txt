cmake_minimum_required(VERSION 3.0)
link_directories("/usr/local/lib")
find_package(Ceres CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

add_executable(optical_flow optical_flow.cpp)
target_link_libraries(optical_flow PRIVATE ${OpenCV_LIBS} fmt::fmt Eigen3::Eigen)
