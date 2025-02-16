cmake_minimum_required(VERSION 3.30)
project(openmp_task)

set(CMAKE_CXX_STANDARD 23)

# Set OpenCV_DIR to the location of OpenCVConfig.cmake
set(ENV{OPENCV_DIR} )
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

# Add OpenMP if available
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# Add the executable and link OpenCV libraries
add_executable(openmp_task main.cpp)
target_link_libraries(openmp_task ${OpenCV_LIBS} ${OpenMP_CXX_LIBRARIES})
