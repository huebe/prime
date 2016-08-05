cmake_minimum_required(VERSION 3.5)
project(prime_c)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")

set(SOURCE_FILES main.c)
add_executable(prime_c ${SOURCE_FILES})