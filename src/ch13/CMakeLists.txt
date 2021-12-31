
cmake_minimum_required(VERSION 3.0)
project(slam_project)
find_package(spdlog CONFIG REQUIRED)
set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        ${Pangolin_LIBRARIES} 
        Eigen3::Eigen
        spdlog::spdlog
        )

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)