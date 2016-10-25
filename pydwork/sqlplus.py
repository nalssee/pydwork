"""
SQLite3 based utils for statistical analysis

Reeling off rows from db(SQLite3) and saving them back to db
"""

import os
import sys
import csv
import re
import sqlite3
import tempfile

from collections import Counter, OrderedDict
from contextlib import contextmanager
from itertools import groupby, islice

import pandas as pd
import numpy as np
import statsmodels.api as sm

from .util import isnum, istext, yyyymm, yyyymmdd, \
                  listify, camel2snake, peek_first


__all__ = ['dbopen', 'Row', 'Rows', 'set_workspace']


WORKSPACE = ''


class Row:
    """
    Mutable version of sqlite3.Row
    It is not safe but essential I suppose.
    """
    # Python 3.6 is expected to use an ordered dict for keyword args
    # If so, we may consider passing kwargs
    def __init__(self):
        super().__setattr__('_ordered_dict', OrderedDict())

    @property
    def columns(self):
        "List[str]: column names"
        return list(self._ordered_dict.keys())

    @property
    def values(self):
        "List[type]"
        return list(self._ordered_dict.values())

    def __getattr__(self, name):
        return self._ordered_dict[name]

    def __setattr__(self, name, value):
        self._ordered_dict[name] = value

    def __delattr__(self, name):
        del self._ordered_dict[name]

    def __getitem__(self, name):
        return self._ordered_dict[name]

    def __setitem__(self, name, value):
        self._ordered_dict[name] = value

    def __delitem__(self, name):
        del self._ordered_dict[name]

    def __str__(self):
        content = ' | '.join(c + ': ' + str(v) for c, v in \
                             zip(self.columns, self.values))
        return '[' + content + ']'

    # for pickling
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    # TODO
    # hasattr doesn't for properly
    # You can't make it work by changing getters and setters
    # to an ordinary way. But it is slower


class Rows(list):
    """
    a shallow wrapper of a list of row instances """
    # Don't try to define __getattr__, __setattr__
    # List objects has a lot of useful attributes that can't be overwritten
    # Not the same situation as 'Row' class

    def __getitem__(self, cols):
        "cols: integer or list of strings or comma separated string"
        if isinstance(cols, int):
            return super().__getitem__(cols)
        elif isinstance(cols, slice):
            # keep it as Rows
            return Rows(super().__getitem__(cols))

        cols = listify(cols)
        if len(cols) == 1:
            col = cols[0]
            return [r[col] for r in self]
        else:
            return [[r[c] for c in cols] for r in self]

    def __setitem__(self, cols, vals):
        """vals can be just a list or a list of lists,
        demensions must match
        """
        if isinstance(cols, int) or isinstance(cols, slice):
            return super().__setitem__(cols, vals)

        cols = listify(cols)
        ncols = len(cols)

        # validity check,
        if len(self) != len(vals):
            raise ValueError('Number of values to assign inappropriate')

        # vals must be rectangular!
        if ncols > 1:
            for vs in vals:
                if len(vs) != ncols:
                    raise ValueError('Invalid values to assign', vs)

        if ncols == 1:
            col = cols[0]
            for r, v in zip(self, vals):
                r[col] = v
        else:
            for r, vs in zip(self, vals):
                for c, v in zip(cols, vs):
                    r[c] = v

    def __delitem__(self, cols):
        if isinstance(cols, int) or isinstance(cols, slice):
            return super().__delitem__(cols)

        cols = listify(cols)
        ncols = len(cols)

        if ncols == 1:
            col = cols[0]
            for r in self:
                del r[col]
        else:
            for r in self:
                for c in cols:
                    del r[c]

    def order(self, key, reverse=0):
        key = _build_keyfn(key)
        self.sort(key=key, reverse=reverse)
        return self

    def where(self, pred):
        pred = _build_keyfn(pred)
        return Rows(r for r in self if pred(r))

    def num(self, cols):
        "another simplified filtering, numbers only"
        cols = listify(cols)
        return self.where(lambda r: all(isnum(r[col]) for col in cols))

    def text(self, cols):
        "another simplified filtering, texts(string) only"
        cols = listify(cols)
        return self.where(lambda r: all(istext(r[col]) for col in cols))

    def ols(self, model):
        left, right = model.split('~')
        yvar = left.strip()
        xvars = [x.strip() for x in right.split('+')]
        Y = self[yvar]
        X = sm.add_constant(self[xvars])
        return sm.OLS(Y, X).fit()

    def truncate(self, col, limit=0.01):
        "Truncate extreme values, defalut 1 percent on both sides"
        xs = self[col]
        lower = np.percentile(xs, limit * 100)
        higher = np.percentile(xs, (1 - limit) * 100)
        return self.where(lambda r: r[col] >= lower and r[col] <= higher)

    def group(self, key):
        yield from _gby(self, key)

    def show(self, n=30, cols=None):
        if self == []:
            print(self)
        else:
            _show(self, n, cols, None)

    def describe(self, n=30, cols=None, percentile=None):
        if self == []:
            print(self)
        else:
            _describe(self, n, cols, percentile)

    # Simpler version of show (when you write it to a file)
    def write(self, filename, cols=None):
        _show(self, None, cols, filename)

    # Use this when you need to see what's inside
    # for example, when you want to see the distribution of data.
    def df(self, cols=None):
        if cols:
            cols = listify(cols)
            return pd.DataFrame([[r[col] for col in cols] for r in self],
                                columns=cols)
        else:
            cols = self[0].columns
            seq = _safe_values(self, cols)
            return pd.DataFrame(list(seq), columns=cols)


class SQLPlus:
    """
    Attributes:
        tables (List[str]): list of all tables in the DB
    """

    def __init__(self, dbfile):
        """
        Args:
            dbfile (str): db filename or ':memory:'
        """
        if dbfile != ':memory:':
            dbfile = os.path.join(WORKSPACE, dbfile)
        self.conn = sqlite3.connect(dbfile)
        self._cursor = self.conn.cursor()
        self.tables = self._list_tables()

        # load some user-defined functions from helpers.py
        self.conn.create_function('isnum', 1, isnum)
        self.conn.create_function('istext', 1, istext)
        self.conn.create_function('yyyymm', 2, yyyymm)
        self.conn.create_function('yyyymmdd', 3, yyyymmdd)


    # args can be a list, a tuple or a dictionary
    def run(self, query, args=()):
        """Simply executes sql statement and update tables attribute

        query: SQL query string
        args: args for SQL query
        """
        self._cursor.execute(query, args)
        self.tables = self._list_tables()

    def reel(self, query, group=False, args=()):
        """Generates a sequence of rows from a query.

        query:  select statement or table name
        """
        qrows = self._cursor.execute(_select_statement(query, '*'), args)
        columns = [c[0] for c in qrows.description]
        # there can't be duplicates in column names
        if len(columns) != len(set(columns)):
            raise ValueError('duplicates in columns names')

        if group:
            yield from _gby(_build_rows(qrows, columns), group)
        else:
            yield from _build_rows(qrows, columns)

    def rows(self, query, args=()):
        "Returns a 'Rows' instance"
        return Rows(self.reel(query, args))

    def df(self, query, cols=None, args=()):
        return self.rows(query, args=args).df(cols)

    def save(self, x, name=None, fn=None, args=()):
        """create a table from an iterator.

        CAUTION!! OVERWIRTES A TABLE IF EXISTS

        x (str or iter or GF[* -> Row])
        name (str): table name in DB
        fn: function that takes a row(all elements are strings)
            and returns a row, used for csv file transformation
        """
        name1, rows = _x2rows(x, self._cursor, args)
        name = name or name1
        if not name:
            raise ValueError('table name required')

        # always overwrites
        self.drop(name)

        rows1 = (fn(r) for r in rows) if fn else rows

        row0, rows2 = peek_first(rows1)
        cols = row0.columns
        seq_values = _safe_values(rows2, cols)

        # You can't save the iterator directly because
        # once you execute a table creation query,
        # then the query in action is changed to the most recent query,
        # not the query for the iterator anymore.

        # You can see the example at test/sqlplus_test.py
        # 'test_run_over_run'

        # So you save the iterator up in another query and reel off it
        with tempfile.NamedTemporaryFile() as f:
            conn = sqlite3.connect(f.name)
            cursor = conn.cursor()

            _sqlite3_save(cursor, seq_values, name, cols)
            _sqlite3_save(self._cursor, _sqlite3_reel(cursor, name, cols),
                          name, cols)
            # no need to commit and close the connection,
            # it's going to be erased anyway

        self.tables = self._list_tables()

    # Be careful so that you don't overwrite the file
    def show(self, x, n=30, cols=None, args=()):
        "Printing to a screen or saving to a file "
        _, rows = _x2rows(x, self._cursor, args)
        _show(rows, n, cols, None)

    def describe(self, query, n=30, cols=None, percentile=None, args=()):
        "Summary"
        _, rows = _x2rows(query, self._cursor, args)
        _describe(rows, n, cols, percentile)

    def write(self, x, filename=None, cols=None, args=()):
        "writes to a file(csv)"
        name, rows = _x2rows(x, self._cursor, args)
        filename = filename or name
        _show(rows, None, cols, filename)

    def drop(self, tables):
        " drop table if exists "
        tables = listify(tables)
        for table in tables:
            # you can't use '?' for table name
            # '?' is for data insertion
            self.run('drop table if exists %s' % table)
        self.tables = self._list_tables()

    def _list_tables(self):
        "List of table names in the database "
        query = self._cursor.execute("""
        select * from sqlite_master
        where type='table'
        """)
        # **.lower()
        tables = [row[1].lower() for row in query]
        return sorted(tables)


@contextmanager
def dbopen(dbfile):
    "Connects to SQL database(sqlite)"
    splus = SQLPlus(dbfile)
    try:
        yield splus
    finally:
        splus.conn.commit()
        splus.conn.close()


def set_workspace(path):
    "all the files and dbs are saved in a given path"
    global WORKSPACE
    WORKSPACE = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)


def _x2rows(x, cursor, args):
    """
    x can be either a string or a generator
    if it is a string it can be either a csv file name or a sql statement

    returns an appropriate name and an iterator of rows
    """
    if isinstance(x, str):
        # csv file name
        if x.endswith('.csv'):
            name = x.split('.')[0].strip()
            return name, _csv_reel(x)
        # sql statement
        else:
            seq_rvals = cursor.execute(_select_statement(x, '*'), args)
            colnames = [c[0] for c in seq_rvals.description]
            name = _starts_with_table_name(x)
            return name, _build_rows(seq_rvals, colnames)
    # if it's a generator
    elif hasattr(x, '__call__'):
        return x.__name__, x(*args)
    # x is an iterable then
    else:
        return None, x


# EVERY COLUMN IS A STRING!!!
def _csv_reel(csv_file):
    "Loads well-formed csv file, 1 header line and the rest is data "
    def is_empty_line(line):
        """Tests if a list of strings is empty for example ["", ""] or []
        """
        return [x for x in line if x.strip() != ""] == []

    if not csv_file.endswith('.csv'):
        csv_file += '.csv'
    with open(os.path.join(WORKSPACE, csv_file)) as fin:
        first_line = fin.readline()[:-1]
        columns = _gen_valid_column_names(listify(first_line))
        ncol = len(columns)

        for line_no, line in enumerate(csv.reader(fin), 2):
            if len(line) != ncol:
                if is_empty_line(line):
                    continue
                raise ValueError(
                    """%s at line %s column count not matched %s != %s: %s
                    """ % (csv_file, line_no, ncol, len(line), line))
            row1 = Row()
            for col, val in zip(columns, line):
                row1[col] = val
            yield row1


def _safe_values(rows, cols):
    "assert all rows have cols"
    for r in rows:
        assert r.columns == cols, str(r)
        yield r.values


def _pick(cols, seq):
    " pick only cols for a seq, similar to sql select "
    cols = listify(cols)
    for r in seq:
        r1 = Row()
        for c in cols:
            r1[c] = r[c]
        yield r1


def _gby(seq, key):
    """Group the iterator by a key
    key is like a key function in sort
    """
    key = _build_keyfn(key)
    for _, rs in groupby(seq, key):
        # to list or not to list
        yield Rows(rs)


def _build_keyfn(key):
    " if key is a string return a key function "
    # if the key is already a function, just return it
    if hasattr(key, '__call__'):
        return key

    colnames = listify(key)
    if len(colnames) == 1:
        return lambda r: r[colnames[0]]
    else:
        return lambda r: [r[colname] for colname in colnames]


def _gen_valid_column_names(columns):
    """Generate valid column names from arbitrary ones

    Note:
        Every column name is lowercased
        >>> _gen_valid_column_names(['a', '_b', 'a', 'a1"*c', 'a1c'])
        ['a0', 'a_b', 'a1', 'a1c0', 'a1c1']
    """
    # Some of the sqlite keywords are not allowed for column names
    # http://www.sqlite.org/sessions/lang_keywords.html
    sqlite_keywords = {
        "ABORT", "ACTION", "ADD", "AFTER", "ALL", "ALTER", "ANALYZE", "AND",
        "AS", "ASC", "ATTACH", "AUTOINCREMENT", "BEFORE", "BEGIN", "BETWEEN",
        "BY", "CASCADE", "CASE", "CAST", "CHECK", "COLLATE", "COLUMN",
        "COMMIT", "CONFLICT", "CONSTRAINT", "CREATE", "CROSS", "CURRENT_DATE",
        "CURRENT_TIME", "CURRENT_TIMESTAMP", "DATABASE", "DEFAULT",
        "DEFERRABLE", "DEFERRED", "DELETE", "DESC", "DETACH", "DISTINCT",
        "DROP", "EACH", "ELSE",
        "END", "ESCAPE", "EXCEPT", "EXCLUSIVE", "EXISTS", "EXPLAIN", "FAIL",
        "FOR", "FOREIGN", "FROM", "FULL", "GLOB", "GROUP", "HAVING", "IF",
        "IGNORE", "IMMEDIATE", "IN", "INDEX", "INDEXED", "INITIALLY", "INNER",
        "INSERT", "INSTEAD", "INTERSECT", "INTO", "IS", "ISNULL", "JOIN",
        "KEY", "LEFT", "LIKE", "LIMIT", "MATCH", "NATURAL",
        # no is ok somehow
        # no idea why
        # "NO",
        "NOT", "NOTNULL", "NULL", "OF", "OFFSET", "ON", "OR", "ORDER", "OUTER",
        "PLAN", "PRAGMA", "PRIMARY", "QUERY", "RAISE", "REFERENCES",
        "REGEXP", "REINDEX", "RENAME", "REPLACE", "RESTRICT", "RIGHT",
        "ROLLBACK", "ROW", "SAVEPOINT", "SELECT", "SET", "TABLE", "TEMP",
        "TEMPORARY", "THEN", "TO", "TRANSACTION",
        "TRIGGER", "UNION", "UNIQUE", "UPDATE", "USING", "VACUUM", "VALUES",
        "VIEW", "VIRTUAL", "WHEN", "WHERE",

        # These are not sqlite keywords but attribute names of Row class
        'COLUMNS', 'VALUES',
    }

    default_column_name = 'col'
    temp_columns = []
    for col in columns:
        # save only alphanumeric and underscore
        # and remove all the others
        newcol = camel2snake(re.sub(r'[^\w]+', '', col))
        if newcol == '':
            newcol = default_column_name
        elif not newcol[0].isalpha() or newcol.upper() in sqlite_keywords:
            newcol = 'a_' + newcol
        temp_columns.append(newcol)

    # no duplicates
    if len(temp_columns) == len(set(temp_columns)):
        return temp_columns

    # Tag numbers to column-names starting from 0 if there are duplicates
    cnt = {col: n for col, n in Counter(temp_columns).items() if n > 1}
    cnt_copy = dict(cnt)

    result_columns = []
    for col in temp_columns:
        if col in cnt:
            result_columns.append(col + str(cnt_copy[col] - cnt[col]))
            cnt[col] -= 1
        else:
            result_columns.append(col)
    return result_columns


def _create_statement(name, colnames):
    """create table if not exists foo (...)

    Note:
        Every type is numeric.
        Table name and column names are all lower cased
    """
    schema = ', '.join([col.lower() + ' ' + 'numeric' for col in colnames])
    return "create table if not exists %s (%s)" % (name.lower(), schema)


def _insert_statement(name, ncol):
    """insert into foo values (?, ?, ?, ...)
    Note:
        Column name is lower cased

    ncol : number of columns
    """
    qmarks = ', '.join(['?'] * ncol)
    return "insert into %s values (%s)" % (name.lower(), qmarks)


def _starts_with_table_name(query):
    first_word = query.strip().split(' ')[0]
    if first_word != 'select' and not first_word.endswith('.csv'):
        return first_word
    else:
        return False


def _select_statement(query, cols):
    "turn it to a select stmt "
    if _starts_with_table_name(query):
        return "select %s from %s" % (', '.join(listify(cols)), query)
    return query


def _sqlite3_reel(cursor, table_name, column_names):
    "generates instances of sqlite3.Row"
    q = _select_statement(table_name, column_names)
    yield from cursor.execute(q)


def _sqlite3_save(cursor, srows, table_name, column_names):
    "saves sqlite3.Row instances to db"
    cursor.execute(_create_statement(table_name, column_names))
    istmt = _insert_statement(table_name, len(column_names))
    for r in srows:
        cursor.execute(istmt, r)


def _show(rows, n, cols, filename):
    """Printing to a screen or saving to a file

    rows: iterator of Row instances
    n: maximum number of lines to show
    cols:  columns to show
    """
    # so that you can easily maintain code
    # Searching nrows is easier than searching n in editors
    nrows = n

    if cols:
        rows = _pick(cols, rows)

    row0, rows1 = peek_first(rows)
    cols = row0.columns
    seq_values = _safe_values(rows1, cols)

    # write to a file
    if filename:
        # practically infinite number
        nrows = nrows or sys.maxsize

        if not filename.endswith('.csv'):
            filename = filename + '.csv'

        with open(os.path.join(WORKSPACE, filename), 'w') as fout:
            w = csv.writer(fout)
            w.writerow(cols)
            for vs in islice(seq_values, nrows):
                w.writerow(vs)

    # write to stdout
    else:
        # show practically all columns.
        with pd.option_context("display.max_rows", nrows), \
                pd.option_context("display.max_columns", 1000):
            # make use of pandas DataFrame displaying
            # islice 1 more rows than required
            # to see if there are more rows left
            list_values = list(islice(seq_values, nrows + 1))
            print(pd.DataFrame(list_values[:nrows], columns=cols))
            if len(list_values) > nrows:
                print("...more rows...")


def _describe(rows, n, cols, percentile):
    print('Table Description')
    print('-----------------')
    print('Sample Rows')

    rows1 = Rows(rows)
    _show(rows1, n, cols, None)

    percentile = percentile if percentile else \
        [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    df = rows1.df(cols)
    print('-----------------')
    print()
    print('Summary Stats')
    print(df.describe(percentile, include='all'))
    print()
    print('Corr Matrix')
    print('-----------')
    print(df.corr())
    print()


# sequence of row values to rows
def _build_rows(seq_values, cols):
    "build rows from an iterator of values"
    for vals in seq_values:
        r = Row()
        for col, val in zip(cols, vals):
            r[col] = val
        yield r
