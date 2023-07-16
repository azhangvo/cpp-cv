cmake_minimum_required(VERSION 3.16)
project(videostreamserver)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig)
pkg_check_modules(PC_ZeroMQ libzmq)

find_path(ZeroMQ_INCLUDE_DIR
        NAMES zmq.hpp
        PATHS ${PC_ZeroMQ_INCLUDE_DIRS}
        )
message("${ZeroMQ_INCLUDE_DIR}")
find_library(ZeroMQ_LIBRARY
        NAMES zmq
        PATHS ${PC_ZeroMQ_LIBRARY_DIRS}
        )
message("${ZeroMQ_LIBRARY}")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)

add_executable(videostreamserver main.cpp)

target_include_directories(videostreamserver PUBLIC ${ZeroMQ_INCLUDE_DIR})

target_link_libraries(videostreamserver ${ZeroMQ_LIBRARY})
target_link_libraries(videostreamserver Eigen3::Eigen)
target_link_libraries(videostreamserver ${OpenCV_LIBS})