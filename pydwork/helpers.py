"""
User-defined functions for sqlite3
"""


from datetime import datetime
from dateutil.relativedelta import relativedelta


# If the return value is True it is converted to 1 or 0 in sqlite3
def isnum(x):
    return isinstance(x, float) or isinstance(x, int)


def istext(x):
    return isinstance(x, str)


def yyyymm(date, n):
    d1 = datetime.strptime(str(date), '%Y%m') + relativedelta(months=n)
    return int(d1.strftime('%Y%m'))

