cmake_minimum_required(VERSION 3.0)

project(optimization)

find_package(Ceres CONFIG REQUIRED)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
add_executable(newton gauss_newton.cpp)
target_link_libraries(newton ${OpenCV_LIBS} fmt::fmt Eigen3::Eigen ${MATPLOTLIB_CPP_INCLUDE_DIRS} ${PYTHON_LIBRARIES})
add_executable(ceres_fitting ceres_fitting.cpp)
target_link_libraries(ceres_fitting Ceres::ceres ${OpenCV_LIBS})


add_executable(learn_ceres basic_ceres.cpp)
target_link_libraries(learn_ceres ceres Ceres::ceres ${OpenCV_LIBS} fmt::fmt)