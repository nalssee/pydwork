from pydwork.npc import *
from pydwork.mypy import timeit
import unittest


class CnPUtilsTest(unittest.TestCase):
    # This is not a proper unitest though.
    def test_npc(self):

        def fib(n):
            if n < 2:
                return n
            else:
                return fib(n - 1) + fib(n - 2)

        fib_producers = producers(fib)

        print('It takes a few seconds', end='\n\n')
        input_vals = [34] * 4

        print('Sequential')
        with timeit():
            print([fib(x) for x in input_vals])
        print('\n')

        print('Parallel')
        with timeit():
            result = []
            npc(fib_producers(input_vals, 2), lambda x: result.append(x), parallel=True)
            print(result)
            self.assertEqual(len(result), len(input_vals))
        print('\n')

        print('Threading')
        with timeit():
            result = []
            npc(fib_producers(input_vals, 2), lambda x: result.append(x))
            print(result)
            self.assertEqual(len(result), len(input_vals))


if __name__ == '__main__':
    unittest.main()
