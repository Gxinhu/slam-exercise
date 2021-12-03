cmake_minimum_required(VERSION 3.0)
set(CMAKE_TOOLCHAIN_FILE /workspace/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file")
project(learning_cmake)
find_package(OpenCV CONFIG REQUIRED)
    # Note: 15 target(s) were omitted.
add_executable(DisplayImage DisplayImage.cpp)
target_link_libraries(DisplayImage ${OpenCV_LIBS}) 
