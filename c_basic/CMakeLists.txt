cmake_minimum_required(VERSION 3.5)
project(prime_c)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(prime_c ${SOURCE_FILES})