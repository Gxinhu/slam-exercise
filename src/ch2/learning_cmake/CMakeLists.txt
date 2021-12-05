cmake_minimum_required(VERSION 3.0)
project(learning_cmake)
    # Note: 15 target(s) were omitted.
add_executable(DisplayImage DisplayImage.cpp)
target_link_libraries(DisplayImage ${OpenCV_LIBS}) 
