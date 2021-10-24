cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(coms527-project)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories("${TORCH_INCLUDE_DIRS}")

find_package(MPI REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_OPTIONS}")

include_directories("${MPI_CXX_INCLUDE_DIRS}")

enable_testing()

add_subdirectory(src)
