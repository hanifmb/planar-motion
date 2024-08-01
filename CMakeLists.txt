cmake_minimum_required(VERSION 3.5)

add_definitions(-DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)

project(planar-motion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${CMAKE_SOURCE_DIR}/include )

add_executable(outlier_rejection example/outlier_rejection.cpp)
add_executable(2pt_ransac example/2pt_ransac.cpp)
add_executable(2pt_direct example/2pt_direct.cpp)
target_link_libraries( outlier_rejection ${OpenCV_LIBS} )
target_link_libraries( 2pt_ransac ${OpenCV_LIBS} )
target_link_libraries( 2pt_direct ${OpenCV_LIBS} )
