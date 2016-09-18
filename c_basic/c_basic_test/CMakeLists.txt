cmake_minimum_required(VERSION 3.5)

project(c_basic_test)

# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Link runTests with what we want to test and the GTest and pthread library
add_executable(c_basic_test bitfunctions_test.cpp)
target_link_libraries(c_basic_test ${GTEST_LIBRARIES} pthread)