import contextlib
import time


@contextlib.contextmanager
def timeit():
    start = time.time()
    try:
        yield
    finally:
        print('\nTotal time: ', time.time() - start)
