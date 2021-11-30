cmake_minimum_required(VERSION 3.0)
set(CMAKE_TOOLCHAIN_FILE /workspace/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file")
project(HelloEigen)
find_package(Eigen3 REQUIRED)
add_executable(eigenMatrix eigenMatrix.cpp)
target_link_libraries(eigenMatrix Eigen3::Eigen) 