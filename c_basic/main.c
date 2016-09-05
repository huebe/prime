#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

typedef int bool;
enum {
  false, true
};

const int cMax = 100000;

void searchPrimesBasic(unsigned int max, bool *isPrime) {
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
  for (int i = 3; i <= max; i += 2) {
    for (int j = 3; j < i / 2 && isPrime[i]; j += 2) {
      if ((i % j) == 0) {
        isPrime[i] = false;
      }
    }
  }
}

void searchPrimes2(unsigned int max, bool *isPrime) {
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
  for (int i = 3; i <= max; i += 2) {
    for (int j = 3; j <= sqrt(i) && isPrime[i]; j += 2) {
      if ((i % j) == 0) {
        isPrime[i] = false;
      }
    }
  }
}

void searchAndPrint(void (*searchPrimes)(unsigned int, bool*), int max) {
  bool *isPrime = (bool *) malloc(max * sizeof(bool));

  clock_t tStart = clock();
  (*searchPrimes)(max, isPrime);
  clock_t tEnd = clock();

  int numPrimeNumbers = 0;
  for (int i = 0; i < max; i++) {
    if (isPrime[i]) {
      printf("%i, ", i);
      numPrimeNumbers++;
    }
  }
  printf("are prime numbers.\nTotal %i prime numbers.\nElapsed: %f seconds\n", numPrimeNumbers, (double)(tEnd - tStart) / CLOCKS_PER_SEC);
  free(isPrime);
}

int main() {
  searchAndPrint(&searchPrimesBasic, cMax);
  searchAndPrint(&searchPrimes2, cMax);
  getchar();


  return 0;
}