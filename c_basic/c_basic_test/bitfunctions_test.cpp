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

TEST(UlonglongBitSetTest, ArrayItemBitSelection) {
  ASSERT_EQ(GET_BITMASK_ULONGLONG(0), 1);
  ASSERT_EQ(GET_BITMASK_ULONGLONG(63), 1ULL << 63);
  ASSERT_EQ(GET_BITMASK_ULONGLONG(64 + 0), 1);
  ASSERT_EQ(GET_BITMASK_ULONGLONG(64 + 63), 1ULL << 63);
}

TEST(UlonglongBitSetTest, ArrayItemSelection) {
  ASSERT_EQ(GET_ARRAY_ULONGLONG(0), 0);
  ASSERT_EQ(GET_ARRAY_ULONGLONG(63), 0);
  ASSERT_EQ(GET_ARRAY_ULONGLONG(64), 1);
  ASSERT_EQ(GET_ARRAY_ULONGLONG(64 + 63 ), 1);
}


TEST(UlonglongBitSetTest, SetBits) {
  unsigned long long testArray[10];
  memset(testArray, 0, sizeof(testArray));
  SET_BIT_ULONGLONG_ARRAY(testArray, 2);
  SET_BIT_ULONGLONG_ARRAY(testArray, 63);
  ASSERT_EQ(testArray[0], 0x8000000000000004);
  SET_BIT_ULONGLONG_ARRAY(testArray, 64 + 2);
  SET_BIT_ULONGLONG_ARRAY(testArray, 64 + 63);
  ASSERT_EQ(testArray[1], 0x8000000000000004);
}

TEST(UlonglongBitSetTest, ClearBits) {
  unsigned long long testArray[10];
  memset(testArray, 0xFF, sizeof(testArray));
  for(int i = 0; i < 2 * ULONGLONG_SIZE_BITS; i++) {
    if(i == 2 || i == 63|| (i == (64 + 2)) || (i == (64 + 63))) // set the trap
      continue;
    CLEAR_BIT_ULONGLONG_ARRAY(testArray, i);
  }
  ASSERT_EQ(testArray[0], 0x8000000000000004);
  ASSERT_EQ(testArray[1], 0x8000000000000004);
}

TEST(UlonglongBitSetTest, ReadBits) {
  unsigned long long testArray[10];
  memset(testArray, 0xFF, sizeof(testArray));
  for(int i = 0; i < 2 * ULONGLONG_SIZE_BITS; i++) {
    if(i == 2 || i == 63|| (i == (64 + 2)) || (i == (64 + 63))) // set the trap
      continue;
    CLEAR_BIT_ULONGLONG_ARRAY(testArray, i);
  }
  ASSERT_EQ(0, READ_BIT_ULONGLONG_ARRAY(testArray, 0));
  ASSERT_EQ(1, READ_BIT_ULONGLONG_ARRAY(testArray, 2));
  ASSERT_EQ(1, READ_BIT_ULONGLONG_ARRAY(testArray, 63));
  ASSERT_EQ(0, READ_BIT_ULONGLONG_ARRAY(testArray, 0 + 64));
  ASSERT_EQ(1, READ_BIT_ULONGLONG_ARRAY(testArray, 2 + 64));
  ASSERT_EQ(1, READ_BIT_ULONGLONG_ARRAY(testArray, 63 + 64));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
