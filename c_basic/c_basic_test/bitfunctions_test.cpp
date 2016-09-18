//
// Created by ch1pmunk on 18.09.16.
//
#include "../c_basic/bitfunctions.h"
#include <gtest/gtest.h>

TEST(UlonglongBitSetTest, Is64BitsPlatform) {
  printf("unsigned long long sizeof: %lu bytes\n", ULONGLONG_SIZE);
  printf("unsigned long long bits count: %lu bits\n", ULONGLONG_SIZE_BITS);
  ASSERT_EQ(ULONGLONG_SIZE, 8);
  ASSERT_EQ(ULONGLONG_SIZE_BITS, 64);
}

TEST(UlonglongBitSetTest, SetBits) {
  unsigned long long testArray[10];
  memset(testArray, 0, sizeof(testArray));
  SET_BIT_ULONGLONG_ARRAY(testArray, 2);
  SET_BIT_ULONGLONG_ARRAY(testArray, 63);
  ASSERT_EQ(testArray[0], 0x8000000000000004);
}



int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
