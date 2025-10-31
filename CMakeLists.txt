cmake_minimum_required(VERSION 3.20)
project(custom_allocator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 1. Add the path to our custom module and include it.
# This script will find Torch, detect CUDA/HIP, and set up the build environment.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CrossTorchGPU)

# Now, variables like TORCH_FOUND and CROSS_BACKEND are available.
message(STATUS "Project using Torch backend: ${CROSS_BACKEND}")

set(CPM_DOWNLOAD_LOCATION ${CMAKE_BINARY_DIR}/CPM.cmake)
if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
  file(DOWNLOAD
    https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/cpm.cmake
    ${CPM_DOWNLOAD_LOCATION}
  )
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development) 

include(${CPM_DOWNLOAD_LOCATION})
CPMAddPackage("gh:pybind/pybind11#v2.13.1")

add_subdirectory(memory)
add_subdirectory(IO)
