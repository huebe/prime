//
// Created by huebe on 20.09.16.
//
#include <omp.h>
#include <math.h>
#include "bitfunctions.h"

void searchPrimesBasicOMP(unsigned long long max, unsigned long long *isPrime) {
  //sanity checks and init stuff
  if (max == 0) return;
  CLEAR_BIT_ULONGLONG_ARRAY(isPrime, 0);
  CLEAR_BIT_ULONGLONG_ARRAY(isPrime, 1);
  if (max < 2) return;
  SET_BIT_ULONGLONG_ARRAY(isPrime, 2);
  SET_BIT_ULONGLONG_ARRAY(isPrime, 3);
  for (int i = 4; i <= max; ((i % 2 == 1) ? SET_BIT_ULONGLONG_ARRAY(isPrime, i) : CLEAR_BIT_ULONGLONG_ARRAY(isPrime, i)), i++);

#pragma omp parallel for schedule(static, 100000)
  for (unsigned long long i = 3; i <= max; i += 2) {
    for (unsigned long long j = 3; j <= sqrt(i) && READ_BIT_ULONGLONG_ARRAY(isPrime, i); j += 2) {
      if ((i % j) == 0) {
#pragma omp critical
        CLEAR_BIT_ULONGLONG_ARRAY(isPrime, i);
      }
    }
  }
}