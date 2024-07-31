cmake_minimum_required(VERSION 3.5)

add_definitions(-DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)

project(planar-motion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${CMAKE_SOURCE_DIR}/include )
include_directories( ${CMAKE_SOURCE_DIR}/include/eigen-3.4.0 )

#add_executable(planar-motion main.cpp)
add_executable(fmat_2pts example/fmat_2pts.cpp)
add_executable(fmat_2pts_ransac example/fmat_2pts_ransac.cpp)
add_executable(draw_x example/draw_x.cpp)
add_executable(good_features example/good_features.cpp)
add_executable(2pts_cvmat example/2pts_cvmat.cpp)
target_link_libraries( fmat_2pts ${OpenCV_LIBS} )
target_link_libraries( fmat_2pts_ransac ${OpenCV_LIBS} )
target_link_libraries( draw_x ${OpenCV_LIBS} )
target_link_libraries( good_features ${OpenCV_LIBS} )
target_link_libraries( 2pts_cvmat ${OpenCV_LIBS} )
#target_link_libraries( planar-motion ${OpenCV_LIBS} )
