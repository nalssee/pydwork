import concurrent.futures
import math

PRIMES = iter([
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419])


def is_prime(n):
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def main():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        PRIMES = iter([
            112272535095293,
            112582705942171,
            112272535095293,
            115280095190773,
            115797848077099,
            1099726899285419])


       
        with open('foo', 'w') as f:      
            result = []
            for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
                result.append([number, prime])
            f.write(str(result))


if __name__ == '__main__':
    main()




