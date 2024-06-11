cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cuda_kernels LANGUAGES C CXX CUDA)

# Prohibit in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source build are not supported")
endif()

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

include(scripts/cmake/generic.cmake)

include_directories(include)
add_subdirectory(src)