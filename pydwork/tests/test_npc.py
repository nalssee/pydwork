import os
import sys
import time
import unittest

TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)

from pydwork.npc import *

import contextlib

@contextlib.contextmanager
def timeit():
    start = time.time()
    try:
        yield
    finally:
        print('\nTotal time: ', time.time() - start)


class CnPUtilsTest(unittest.TestCase):
    # This is not a proper unitest though.
    def test_npc(self):

        def fib(n):
            if n < 2:
                return n
            else:
                return fib(n - 1) + fib(n - 2)

        print('It takes a few seconds', end='\n\n')
        input_vals = [34] * 4

        print('Sequential')
        with timeit():
            print([fib(x) for x in input_vals])
        print('\n')

        print('Parallel')
        with timeit():
            result = []
            npc(make_producers(fib, input_vals, 2), lambda x: result.append(x),
                parallel=True)
            print(result)
            self.assertEqual(len(result), len(input_vals))
        print('\n')

        print('Threading')
        with timeit():
            result = []
            npc(make_producers(fib, input_vals, 2), lambda x: result.append(x))
            print(result)
            self.assertEqual(len(result), len(input_vals))


if __name__ == '__main__':
    unittest.main()
