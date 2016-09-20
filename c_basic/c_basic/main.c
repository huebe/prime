#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "bitfunctions.h"

const unsigned long long cMax = 1000;

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

/* void searchPrimes2(unsigned long long max, bool *isPrime) {
  //sanity checks and init stuff
  if (max == 0) return;
  isPrime[0] = false;
  isPrime[1] = false;
  if (max < 2) return;
  isPrime[2] = true; //special case
  isPrime[3] = true;
  for (int i = 4; i <= max;
       isPrime[i] = (i % 2 == 1) ? true : false, i++); //even numbers > 2 are NEVER prime numbers, ods can be

  //lets search for prime numbers
  for (unsigned long long i = 3; i <= max; i += 2) {
    for (unsigned long long j = 3; j <= sqrt(i) && isPrime[i]; j += 2) {
      if ((i % j) == 0) {
        isPrime[i] = false;
      }
    }
  }
} */

void searchAndPrint(void (*searchPrimes)(unsigned long long, unsigned long long*), unsigned long long max) {
  unsigned long long *isPrime = (unsigned long long *) malloc((max / ULONGLONG_SIZE_BITS) + 1);

  clock_t tStart = clock();
  (*searchPrimes)(max, isPrime);
  clock_t tEnd = clock();

  int numPrimeNumbers = 0;
  for (int i = 0; i < max; i++) {
    if (READ_BIT_ULONGLONG_ARRAY(isPrime, i)) {
      printf("%i, ", i);
      numPrimeNumbers++;
    }
  }
  printf("are prime numbers.\nTotal %i prime numbers.\nElapsed: %f seconds\n", numPrimeNumbers, (double)(tEnd - tStart) / CLOCKS_PER_SEC);
  free(isPrime);
}

int main() {
  searchAndPrint(&searchPrimesBasic, cMax);
  //searchAndPrint(&searchPrimes2, cMax);
  getchar();


  return 0;
}