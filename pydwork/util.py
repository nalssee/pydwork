import contextlib
import time
import math


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
