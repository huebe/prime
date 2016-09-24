#if __STDC_VERSION__ >= 199901L
#define _XOPEN_SOURCE 600
#else
#define _XOPEN_SOURCE 500
#endif /* __STDC_VERSION__ */


#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "bitfunctions.h"
#include "prime_omp.h"

const unsigned long long cMax = 1000000;


void searchPrimesBasic(unsigned long long max, unsigned long long *isPrime) {
  //sanity checks and init stuff
  if (max == 0) return;
  CLEAR_BIT_ULONGLONG_ARRAY(isPrime, 0);
  CLEAR_BIT_ULONGLONG_ARRAY(isPrime, 1);
  if (max < 2) return;
  SET_BIT_ULONGLONG_ARRAY(isPrime, 2);
  SET_BIT_ULONGLONG_ARRAY(isPrime, 3);
  for (int i = 4; i <= max; ((i % 2 == 1) ? SET_BIT_ULONGLONG_ARRAY(isPrime, i) : CLEAR_BIT_ULONGLONG_ARRAY(isPrime, i)), i++);

  //lets search for prime numbers
  for (unsigned long long i = 3; i <= max; i += 2) {
    for (unsigned long long j = 3; j < i / 2 && READ_BIT_ULONGLONG_ARRAY(isPrime, i); j += 2) {
      if ((i % j) == 0) {
        CLEAR_BIT_ULONGLONG_ARRAY(isPrime, i);
      }
    }
  }
}

void searchPrimesBasic2(unsigned long long max, unsigned long long *isPrime) {
  //sanity checks and init stuff
  if (max == 0) return;
  CLEAR_BIT_ULONGLONG_ARRAY(isPrime, 0);
  CLEAR_BIT_ULONGLONG_ARRAY(isPrime, 1);
  if (max < 2) return;
  SET_BIT_ULONGLONG_ARRAY(isPrime, 2);
  SET_BIT_ULONGLONG_ARRAY(isPrime, 3);
  for (int i = 4; i <= max; ((i % 2 == 1) ? SET_BIT_ULONGLONG_ARRAY(isPrime, i) : CLEAR_BIT_ULONGLONG_ARRAY(isPrime, i)), i++);

  //lets search for prime numbers
  for (unsigned long long i = 3; i <= max; i += 2) {
    for (unsigned long long j = 3; j <= sqrt(i) && READ_BIT_ULONGLONG_ARRAY(isPrime, i); j += 2) {
      if ((i % j) == 0) {
        CLEAR_BIT_ULONGLONG_ARRAY(isPrime, i);
      }
    }
  }
}


void searchAndPrint(void (*searchPrimes)(unsigned long long, unsigned long long*), unsigned long long max) {
  unsigned long long *isPrime = (unsigned long long *) malloc(((max / ULONGLONG_SIZE_BITS) + 1) * sizeof(unsigned long long));

  struct timespec start, finish;
  double elapsed;
  clock_gettime(CLOCK_MONOTONIC, &start);
  clock_t tStart = clock();
  (*searchPrimes)(max, isPrime);
  clock_t tEnd = clock();
  clock_gettime(CLOCK_MONOTONIC, &finish);
  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;


  int numPrimeNumbers = 0;
  for (int i = 0; i < max; i++) {
    if (READ_BIT_ULONGLONG_ARRAY(isPrime, i)) {
    //  printf("%i, ", i);
      numPrimeNumbers++;
    }
  }

  printf("are prime numbers.\nTotal %i prime numbers.\nPocessor time elapsed: %f seconds\nWall time elapsed: %f seconds\n",
         numPrimeNumbers, (double)(tEnd - tStart) / CLOCKS_PER_SEC, elapsed);
  free(isPrime);
}

int main() {
  searchAndPrint(&searchPrimesBasicOMP, cMax);
  searchAndPrint(&searchPrimesBasic2, cMax);
  return 0;
}