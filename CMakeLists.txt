cmake_minimum_required(VERSION 3.8)
project(intelligent_scissor)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall")

# Deal with OpenCV problem on Mac
set(CMAKE_PREFIX_PATH "/usr/local/Cellar/opencv/3.4.0_1/share/OpenCV")

set(OpenCV_INCLUDE_DIRS "/usr/local/Cellar/opencv/3.4.0_1/include/")
set(OpenCV_LIBS "/usr/local/Cellar/opencv/3.4.0_1/lib/libopencv_core.3.4.dylib")

find_package(OpenCV REQUIRED)
find_package(boost REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(src/inc)

#file(GLOB_RECURSE USER_SOURCES "src/*.cpp")

add_executable(intelligent_scissor src/main.cpp src/fibheap.cpp)

target_link_libraries(intelligent_scissor ${OpenCV_LIBS})