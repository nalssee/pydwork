import random
import string
import re
import fileinput

from datetime import datetime
from dateutil.relativedelta import relativedelta

from itertools import dropwhile, chain, zip_longest

import multiprocessing as mp
import threading as th
from queue import Queue


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


def mpairs(seq1, seq2, key1, key2=None):
    """Generates a tuple of matching pairs
    key1 and key2 are functions (getters, sort of)

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
# and easy to write and also doesn't cause a lot of troubles.
# Nevertheless I want to minimize any nondeterminism at all costs.
def pmap(func, seq,
         chunksize=1, nworkers=mp.cpu_count(),
         fargs=[], parallel=True):
    """
    parallel map, ordered version
    if you are curious about the parameters, refer to multiprocessing library
    documentation (Pool.imap)

    fargs: first args for each 'func'
           unless fargs is not [], 'nworkers' is ignored
           number of fargs becomes nworkers

           You may think of those cases you want to use selenium drivers
           for web scraping. If you have, say, 4 workers you need 4 drivers

    parallel: If parallel is False, wokers are threads, not processes
    """
    if fargs:
        nworkers = len(fargs)
    else:
        fargs = [None] * nworkers

    the_end = random_string()
    create_que = mp.Queue if parallel else Queue
    create_worker = mp.Process if parallel else th.Thread

    # Opening multiple ques sounds dumb in a way
    # but this is a easier way to implement the ordered version of
    # parrallel map. It's just that there is a limit in the number of
    # ques in the OS. Of course you wouldn't make more than 20 processes.
    que1s = [create_que(1) for _ in range(nworkers)]
    que2s = [create_que(1) for _ in range(nworkers)]

    def insert1(seq, que1s):
        for chunks in grouper(grouper(seq, chunksize, the_end),
                              nworkers, the_end):
            for que1, chunk in zip(que1s, chunks):
                que1.put(chunk)
        for que1 in que1s:
            que1.put(the_end)

    w0 = create_worker(target=insert1, args=(seq, que1s))
    w0.daemon = True
    w0.start()

    def insert2(func, que1, que2):
        while True:
            chunk = que1.get()
            if chunk == the_end:
                que2.put(the_end)
                return
            else:
                result = []
                for x in chunk:
                    if x != the_end:
                        try:
                            result.append(func(x))
                        except Exception as error:
                            que2.put(the_end)
                            str_x = str(x)
                            if len(str_x) > 100:
                                str_x = str_x[:80] + ' ... ' + str_x[-20:]
                            print('child worker error: ' + repr(error), str_x)
                            return
                que2.put(result)

    for farg, que1, que2 in zip(fargs, que1s, que2s):
        if farg:
            # passing lexical closure
            # you can just do 'lambda x: func(farg, x)' for parallel version
            # because Python just copies args for each process
            # but it wouldn't work for thread version
            newfunc = (lambda farg: lambda x: func(farg, x))(farg)
        else:
            newfunc = func
        # don't replace the above with something like:
        # newfunc = A if test else B
        # causes a "can't pickle" error, I have no idea why.
        w = create_worker(target=insert2, args=(newfunc, que1, que2))
        w.daemon = True
        w.start()

    while True:
        for que2 in que2s:
            result = que2.get()
            if result == the_end:
                return
            else:
                yield from result

    # all the processes and threads are set to daemon
    # hence no need to terminate them manually
    # I might be wrong in the long run though.


# The following guys are also going to be
# included in "extended sqlite functions" set

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

