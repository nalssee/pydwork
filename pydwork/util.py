import random
import string
import re
import fileinput
import time

from datetime import datetime
from dateutil.relativedelta import relativedelta

from itertools import dropwhile, chain, zip_longest

import multiprocessing as mp


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


def prepend_header(filename, header=None, drop=1):
    """Drop n lines and prepend header

    Args:
        filename (str)
        header (str)
        drop (int)
    """
    for no, line in enumerate(fileinput.input(filename, inplace=True)):
        # it's meaningless to set drop to -1, -2, ...
        if no == 0 and drop == 0:
            if header:
                print(header)
            print(line, end='')
        # replace
        elif no + 1 == drop:
            if header:
                print(header)
        elif no >= drop:
            print(line, end='')
        else:
            # no + 1 < drop
            continue


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


def camel2snake(name):
    """
    Args:
        name (str): camelCase
    Returns:
        str: snake_case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def peek_first(seq):
    """
    Note:
        peeked first item is pushed back to the sequence
    Args:
        seq (Iter[type])
    Returns:
        Tuple(type, Iter[type])
    """
    seq = iter(seq)
    first_item = next(seq)
    return first_item, chain([first_item], seq)


def listify(colstr):
    """A comma or space separated string to a list of strings

    Args:
        colstr (str)
    Returns:
        List[str]

    Example:
        >>> _listify('a b c')
        ['a', 'b', 'c']

        >>> _listify('a, b, c')
        ['a', 'b', 'c']
    """
    if isinstance(colstr, str):
        if ',' in colstr:
            return [x.strip() for x in colstr.split(',')]
        else:
            return [x for x in colstr.split(' ') if x]
    else:
        return colstr


# copied from 'itertools'
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


# Pool.iamp in multiprocessing doesn't allow
# it to pass locally defined functions
# I won't define unordered version although it is somewhat faster
# and easy to write and also doesn't cause a lot of troubles,
# I want to minimize any nondeterminism at all costs.
def pmap(func, seq, chunksize=1, processes=mp.cpu_count()):
    # I don't think keeping low qsize doesn't affect the performance
    max_qsize = processes
    empty_elt = random_string()
    sentinel = random_string()
    que1, que2 = mp.Queue(max_qsize), mp.Queue(max_qsize)

    counter = mp.Value('i', 0)
    ws = []

    def insert1(seq, que1):
        for i, chunk in enumerate(grouper(seq, chunksize, empty_elt)):
            que1.put((i, chunk))
        que1.put((i + 1, sentinel))

    w0 = mp.Process(target=insert1, args=(seq, que1))
    w0.daemon = True
    w0.start()
    ws.append(w0)

    def safe_put(que2, val, i):
        empty_spin = 0
        while True:
            if i == counter.value:
                que2.put(val)
                time.sleep(0.01)
                counter.value += 1
                break
            empty_spin += 1
            if empty_spin > 10:
                time.sleep(0.1)


    def insert2(que1, que2):
        while True:
            i, chunk = que1.get()
            if chunk == sentinel:
                safe_put(que2, sentinel, i)
                return
            else:
                result = []
                for x in chunk:
                    if x != empty_elt:
                        try:
                            result.append(func(x))
                        except Exception as error:
                            print(repr(error))
                            result.append(sentinel)
                safe_put(que2, result, i)

    for _ in range(processes):
        w = mp.Process(target=insert2, args=(que1, que2))
        w.daemon = True
        w.start()
        ws.append(w)

    while True:
        result = que2.get()
        if result == sentinel:
            for w in ws:
                w.terminate()
            return
        else:
            yield from result


# The following guys are also going to be
# included in "extended sqlite functions"
# set

# If the return value is True it is converted to 1 or 0 in sqlite3
def isnum(x):
    return isinstance(x, float) or isinstance(x, int)


def istext(x):
    return isinstance(x, str)


def yyyymm(date, n):
    d1 = datetime.strptime(str(date), '%Y%m') + relativedelta(months=n)
    return int(d1.strftime('%Y%m'))


def yyyymmdd(date, n):
    d1 = datetime.strptime(str(date), '%Y%m%d') + relativedelta(days=n)
    return int(d1.strftime('%Y%m%d'))

