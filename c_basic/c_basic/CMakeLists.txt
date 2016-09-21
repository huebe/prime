cmake_minimum_required(VERSION 2.8)
project(prime_c)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall -Werror -m64")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(SOURCE_FILES main.c bitfunctions.h prime_omp.c prime_omp.h)
add_executable(prime_c ${SOURCE_FILES})

target_link_libraries(prime_c m)
