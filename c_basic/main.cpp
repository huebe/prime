
#include <stdio.h>
#include <cmath>

const int cMax = 2000000000000;


void searchPrimes(int max, bool *isPrime) {
    for (int i = 3; i <= max; i += 2) {
        for (int j = 3; j < i / 2 && isPrime[i]; j += 2) {
            if ((i % j) == 0) {
                isPrime[i] = false;
            }
        }
    }
}

int main() {
    bool cIsPrime[cMax];
    for (int i = 0; i < cMax; cIsPrime[i] = true, i++);
    searchPrimes(cMax, cIsPrime);
    for (int i = 0; i < cMax; i++) {
        if (cIsPrime[i]) {
            printf("%i, ", i);
        }
    }

    printf("are prime numbers");
    getchar();
    return 0;
}