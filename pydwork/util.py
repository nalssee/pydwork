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
    chunksize = int(math.ceil(len(xs) / n))
    for i in range(n):
        yield xs[chunksize * i:chunksize * (i + 1)]
