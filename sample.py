from pydwork.sqlplus import *
from itertools import groupby


xs = [1,1,1,2,2,3,3,4,4,5,5,5,5,1]

def foo():
    for i, rs in groupby(xs):
        yield from list(rs)

for r in foo():
    print(r)
