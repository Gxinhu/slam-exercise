
cmake_minimum_required(VERSION 3.0)
project(loop_closure)
find_package(Ceres CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(fbow CONFIG REQUIRED)
set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.so" )
add_executable(feature_training feature_training.cpp)
target_link_libraries(feature_training PRIVATE ${OpenCV_LIBS} fmt::fmt Eigen3::Eigen Ceres::ceres ${DBoW3_LIBS} )

add_executable(loop loop_closure.cpp)
target_link_libraries(loop PRIVATE ${OpenCV_LIBS} fmt::fmt Eigen3::Eigen Ceres::ceres ${DBoW3_LIBS} )