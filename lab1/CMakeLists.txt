cmake_minimum_required(VERSION 3.24)
project(lab1)

set(CMAKE_CXX_STANDARD 11)

# Add MPI Package to Project
find_package(MPI REQUIRED)

add_executable(lab1 1var.cpp)

target_link_libraries(lab1 PUBLIC MPI::MPI_CXX)

set(CMAKE_CXX_COMPILER "/usr/bin/mpic++")