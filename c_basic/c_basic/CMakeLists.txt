cmake_minimum_required(VERSION 3.5)
project(prime_c)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -m64")

set(SOURCE_FILES main.c bitfunctions.h)
add_executable(prime_c ${SOURCE_FILES})

target_link_libraries(prime_c m)
