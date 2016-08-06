#include <stdio.h>
#include <time.h>
#include <stdlib.h>

typedef int bool;
enum {
    false, true
};

const int cMax = 1000000;


void searchPrimes(unsigned int max, bool *isPrime) {
    //sanity checks and init stuff
    if (max == 0) return;
    isPrime[0] = true;
    isPrime[1] = true;
    if (max < 2) return;
    isPrime[2] = true; //special case
    for (int i = 4; i <= max; isPrime[i] = (i % 2 == 1) ? true : false, i++); //even numbers > 2 are NEVER prime numbers, ods can be

    //lets search for prime numbers
    for (int i = 3; i <= max; i += 2) {
        for (int j = 3; j < i / 2 && isPrime[i]; j += 2) {
            if ((i % j) == 0) {
                isPrime[i] = false;
            }
        }
    }
}

int main() {
    bool *cIsPrime = (bool*)malloc(cMax * sizeof(bool));

    clock_t tStart = clock();
    searchPrimes(cMax, cIsPrime);
    clock_t tEnd = clock();

    for (int i = 0; i < cMax; i++) {
        if (cIsPrime[i]) {
            printf("%i, ", i);
        }
    }

    printf("are prime numbers. Elapsed: %f seconds\n", (double)(tEnd - tStart) / CLOCKS_PER_SEC);

    getchar();

    free(cIsPrime);
    return 0;
}