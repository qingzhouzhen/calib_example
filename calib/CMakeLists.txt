cmake_minimum_required(VERSION 3.20)
project(untitled)

set(CMAKE_CXX_STANDARD 17)

# find pcl
find_package(PCL REQUIRED COMPONENT common io visualization filters features keypoints)
link_directories(${PCL_LIBRARY_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
add_definitions( ${PCL_DEFINITIONS} )

add_executable(untitled calib.cpp)
target_link_libraries(untitled ${PCL_LIBRARIES})

add_library(pose_calib SHARED calib.cpp)
target_link_libraries(pose_calib ${PCL_LIBRARIES})
