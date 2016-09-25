import random
import string
import re

from datetime import datetime
from dateutil.relativedelta import relativedelta

from itertools import dropwhile, chain


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

