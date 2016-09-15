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


def mpairs(seq1, seq2, key=lambda x: x):
    """Generates a tuple of matching pairs
    seq1, seq2 must be sorted before being passed here
    or you will see unexpected results
    """
    while True:
        try:
            s1 = next(seq1)
            s2 = next(seq2)

            a1 = key(s1)
            a2 = key(s2)

            if a1 == a2:
                yield (s1, s2)
            elif a1 < a2:
                seq1 = dropwhile(lambda x: key(x) < a2, seq1)
                # put it back
                seq2 = chain([s2], seq2)
            else:
                seq1 = chain([s1], seq1)
                seq2 = dropwhile(lambda x: key(x) < a1, seq2)

        except StopIteration:
            break


