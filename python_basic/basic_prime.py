import time

def search_prime(max):
    primes = [2]
    for i in xrange(3, max, 2):
        is_prime = True
        for j in xrange(3, i / 2, 2):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)

    return primes

def main():
    start = time.clock()
    primes = search_prime(1000000)
    end = time.clock()
    print primes, "are prime numbers. Elapsed:", (end - start), "seconds"
    print "Amount of prime numbers found:", len(primes)

if __name__ == "__main__":
    main()

