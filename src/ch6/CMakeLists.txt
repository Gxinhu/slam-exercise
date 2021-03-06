cmake_minimum_required(VERSION 3.0)

project(optimization)

find_package(Ceres CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)
add_executable(newton gauss_newton.cpp)
target_link_libraries(newton ${OpenCV_LIBS} fmt::fmt Eigen3::Eigen Matplot++::cimg std::filesystem Matplot++::matplot Matplot++::nodesoup)
add_executable(ceres_fitting ceres_fitting.cpp)
target_link_libraries(ceres_fitting Ceres::ceres ${OpenCV_LIBS})


add_executable(learn_ceres basic_ceres.cpp)
target_link_libraries(learn_ceres ceres Ceres::ceres ${OpenCV_LIBS} fmt::fmt)

add_executable(g2o_fitting g2o_fitting.cpp)
target_link_libraries(g2o_fitting PRIVATE g2o::core ${OpenCV_LIBS})