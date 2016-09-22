import contextlib
import time
import random
import string

from itertools import zip_longest, chain, dropwhile


@contextlib.contextmanager
def timeit():
    start = time.time()
    try:
        yield
    finally:
        print('\nTotal time: ', time.time() - start)


def nchunks(xs, n):
    """Yields n chunks about the same size. """
    xs = list(xs)
    start = 0
    for i in range(n):
        if i + 1 == n:
            # last chunk
            yield xs[start:]
        else:
            chunksize = round(len(xs[start:]) / (n - i))
            yield xs[start:start + chunksize]
            start += chunksize


# Excerpted from Python referece.
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def random_string(nchars=20):
    "Generates a random string of lengh 'n' with alphabets and digits. "
    return ''.join(random.SystemRandom().choice(string.ascii_letters +
                   string.digits) for _ in range(nchars))


def mpairs(seq1, seq2, key1=lambda x: x, key2=None):
    """Generates a tuple of matching pairs
    key1 and key2 are functions

    seq1, seq2 must be sorted before being passed here
        and also each key value(which is returned by key funcs) must be UNIQUE
        otherwise you will see unexpected results
    """
    key2 = key2 or key1

    s1, s2 = next(seq1), next(seq2)
    k1, k2 = key1(s1), key2(s2)

    while True:
        try:
            if k1 == k2:
                yield (s1, s2)
                s1, s2 = next(seq1), next(seq2)
                k1, k2 = key1(s1), key2(s2)
            elif k1 < k2:
                s1 = next(dropwhile(lambda x: key1(x) < k2, seq1))
                k1 = key1(s1)
            else:
                s2 = next(dropwhile(lambda x: key2(x) < k1, seq2))
                k2 = key2(s2)

        except StopIteration:
            break


