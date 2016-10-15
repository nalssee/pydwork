"""
SQLite3 based utils for statistical analysis

--------------------------------------------------------------
What this program does is just reeling off rows from db(SQLite3)
and saving them back to db
--------------------------------------------------------------

Emperical data analysis task is largely composed of two parts,
wrangling data(cleaning up so you can easily handle it)
and applying statistical methodologies to data.

Python is really great for data analysis, since you have
widely used, reliable tools like Pandas, Numpy, Scipy, ...

Pandas is said-to-be a really great for wrangling data.
But it wasn't a very smooth process for me to learn Pandas.
And you have to load all the data in the system memory.
eo you have to figure out a way around when you need to
handle very large data sets.

If you focus only on data wragling, SQL is an amazingly great
tool. Performance, flexibility, reliablity, ease of use, it will
never disappoint you.

But there's one problem in SQL for data analysis. It doesn't
integrate with python functions out of box. And it doesn't provide
sophisticated statistical functions.

Say you want to group stock returns every year
,do some linear regressions with variables of your interests,
and sum up the coefficients from the task.
If you have to do the work using plain SQL,
your life won't be happy anymore.

It would be great if you can integrate Python with SQL.
It is what I'm trying to do here.

This program is not a sophisticated, full-fledged automation system.
I'd rather say it's just a mental framework.

Use SQL to clean up data and group them by certain criteria,
apply statical tools to each group using Python(statmodels, numpy and so on)
and sum up the results.

If you have some basic SQL and Python knowledge,
(You don't have to be an expert)
I believe this is better that using Pandas for some people.
For me, it is.

As for docstring:
    GF[int->int]: generator function type that yields int
                  and takes int as arg, parameters are optional
    FN[str->str]: Ordinary function type
    iter: types.GeneratorType
    Iter[str]: iterator of strings
    file: file object

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


__all__ = ['dbopen', 'Row', 'set_workspace']


WORKSPACE = ''


class Row:
    """
    Basically the same as sqlite3.Row
    it's just that using sqlite3.Row is a bit clunkier.
    r['col'] = 34
    I also want to write as r.col = 34

    And it's better to keep the order of columns
    """
    # Python 3.6 is expected to used ordered dict for keyword args
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


class Rows(list):
    """
    a shallow wrapper of a list of row instances
    """
    # Don't try to define __getattr__, __setattr__
    # List objects has a lot of useful attributes that can't be overwritten
    # Not the same situation as 'Row' class

    def __getitem__(self, cols):
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

    def filter(self, pred):
        pred = _build_keyfn(pred)
        return Rows(r for r in self if pred(r))

    def truncate(self, col, limit=0.01):
        xs = []
        for r in self:
            val = r[col]
            if isnum(val):
                xs.append(val)

        lower = np.percentile(xs, limit * 100)
        higher = np.percentile(xs, (1 - limit) * 100)

        return self.fromto(col, lower, higher)

    def fromto(self, col, beg, end):
        "simplified fitering, inclusive"
        def testfn(r):
            val = r[col]
            return beg <= val and val <= end
        return self.filter(testfn)

    def ge(self, col, beg):
        "greater than or equal to"
        return self.filter(lambda r: beg <= r[col])

    def le(self, col, end):
        "less than or equal to"
        return self.filter(lambda r: r[col] <= end)

    def num(self, cols):
        "another simplified filtering, numbers only"
        cols = listify(cols)
        return self.filter(lambda r: all(isnum(r[col]) for col in cols))

    def text(self, cols):
        "another simplified filtering, texts(string) only"
        cols = listify(cols)
        return self.filter(lambda r: all(istext(r[col]) for col in cols))

    def equals(self, col, val):
        return self.filter(lambda r: r[col] == val)

    def contains(self, col, vals):
        vals = listify(vals)
        return self.filter(lambda r: r[col] in vals)

    def ols(self, model):
        left, right = model.split('~')
        yvar = left.strip()
        xvars = [x.strip() for x in right.split('+')]
        Y = self[yvar]
        X = sm.add_constant(self[xvars])
        return sm.OLS(Y, X).fit()

    def group(self, key):
        yield from _gby(self, key)

    def show(self, n=30, cols=None, filename=None):
        if self == []:
            print(self)
        else:
            _show(self, n=n, cols=cols, filename=filename)

    # Simpler version of show (when you write it to a file)
    def write(self, filename):
        _show(self, n=None, filename=filename)

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
        self.conn.create_function('yyyymmdd', 2, yyyymmdd)

    # args can be a list, a tuple or a dictionary
    def run(self, query, args=()):
        """Simply executes sql statement and update tables attribute

        Args:
            query (str): SQL query string
            args (List[any] or Tuple[any]): args for SQL query
        """
        self._cursor.execute(query, args)
        self.tables = self._list_tables()

    def reel(self, query, group=False, args=()):
        """Generates a sequence of rows from a query.

        Args:
            query (str): select statement or table name

        Yields:
            Row
        """
        qrows = self._cursor.execute(_select_statement(query), args)
        columns = [c[0] for c in qrows.description]
        # there can't be duplicates in column names
        if len(columns) != len(set(columns)):
            raise ValueError('duplicates in columns names')

        if group:
            yield from _gby(_build_rows(qrows, columns), group)
        else:
            yield from _build_rows(qrows, columns)

    def rows(self, query, args=()):
        return Rows(self.reel(query, args))

    def save(self, seq, name=None, fn=None, args=()):
        """create a table from an iterator.

        Note:<%=  %>
            if seq is a generator function and 'name' is not given,
            the function name is going to be the table name.

        Args:
            seq (str or iter or GF[* -> Row])
            name (str): table name in DB
            args (List[type]): args for seq (GF)
            fn (FN[Row -> Row])
        """
        if isinstance(seq, str):
            name = name or seq.split('.')[0].strip()
            seq = _csv_reel(seq)
        # if 'seq' is a generator function, it is executed to make an iterator
        elif hasattr(seq, '__call__'):
            name = name or seq.__name__
            seq = seq(*args)

        if name is None:
            raise ValueError('table name required')

        # table names are case insensitive
        if name.lower() in self.tables:
            return

        if fn:
            seq = (fn(r) for r in seq)

        row0, seq = peek_first(seq)
        colnames = row0.columns

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

            values1 = _safe_values(seq, colnames)
            _sqlite3_save(cursor, values1, name, colnames)
            _sqlite3_save(self._cursor, _sqlite3_reel(cursor, name, colnames),
                          name, colnames)
            # no need to commit and close the connection,
            # it's going to be erased anyway

        self.tables = self._list_tables()

    # Be careful so that you don't overwrite the file
    def show(self, query, n=30, cols=None, filename=None, args=()):
        """Printing to a screen or saving to a file

        Args:
            query (str or Iter[Row] or GF)
            args (List[type] or Tuple[type]): args for query (GF)
            n (int): maximum number of lines to show
            cols (str or List[str]): columns to show
            filename (str): filename to save
        """
        # so that you can easily maintain code
        # Searching nrows is easier than searching n in editors
        nrows = n

        if isinstance(query, str):
            if query.endswith('.csv'):
                rows = _csv_reel(query)
            else:
                seq_rvals = self._cursor.execute(_select_statement(query), args)
                colnames = [c[0] for c in seq_rvals.description]
                rows = _build_rows(seq_rvals, colnames)

        # then query is an iterator of rows, or a list of rows
        # of course it can be just a generator function of rows
        else:
            rows = query
            if hasattr(rows, '__call__'):
                rows = rows(*args)

        _show(rows, n=nrows, cols=cols, filename=filename)

    # Simpler version of show (when you write it to a file)
    # so you make less mistakes.
    def write(self, query, filename=None, args=()):
        """
        Args:
            query (str or Iter[Row] or GF)
            args (List[type] or Tuple[type]): args for query (GF)
            filename (str): filename to save
        """
        if isinstance(query, str) and \
           _is_oneword(query) and filename is None:
            filename = query
        self.show(query, filename=filename, args=args, n=None)

    def drop(self, tables):
        """
        drop table if exists

        Args:
            tables (str or List[str])
        """
        tables = listify(tables)
        for table in tables:
            # you can't use '?' for table name
            # '?' is for data insertion
            self.run('drop table if exists %s' % table)
        self.tables = self._list_tables()

    def _list_tables(self):
        """List of table names in the database

        Returns:
            List[str]
        """
        query = self._cursor.execute("""
        select * from sqlite_master
        where type='table'
        """)
        # **.lower()
        tables = [row[1].lower() for row in query]
        return sorted(tables)


@contextmanager
def dbopen(dbfile):
    """Connects to SQL database(sqlite)

    Args:
        dbfile (str)
    Yields:
        SQLPlus
    """
    splus = SQLPlus(dbfile)
    try:
        yield splus
    finally:
        splus.conn.commit()
        splus.conn.close()


def set_workspace(path):
    """
    Args:
        path (str)
    """
    global WORKSPACE

    WORKSPACE = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)


# EVERY COLUMN IS A STRING!!!
def _csv_reel(csv_file):
    """Loads well-formed csv file, 1 header line and the rest is data

    Args:
        csv_file (str)
    Yields:
        Row
    """
    def is_empty_line(line):
        # Performance is not so important
        # since this function is invoked only when the line is really wierd
        """Tests if a list of strings is empty for example ["", ""] or []
        """
        return [x for x in line if x.strip() != ""] == []

    if not csv_file.endswith('.csv'):
        csv_file += '.csv'
    with open(os.path.join(WORKSPACE, csv_file)) as fin:
        first_line = fin.readline()[:-1]
        columns = _gen_valid_column_names(listify(first_line))
        ncol = len(columns)

        def rows():
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

        yield from rows()


# I don't like the name
def _safe_values(rows, cols):
    for r in rows:
        assert r.columns == cols, str(r)
        yield r.values


def _pick(cols, seq):
    """
    Args:
        cols (str or List[str])
        seq (Iter[Row])
    Yields:
        Row
    """
    cols = listify(cols)
    for r in seq:
        r1 = Row()
        for c in cols:
            r1[c] = r[c]
        yield r1


def _gby(seq, key):
    """Group the iterator by a key

    Args
        seq (iter)
        key (FN[Row -> type] or List[str] or str): if [], group them all
    Yields:
        List[Row]
    """
    key = _build_keyfn(key)
    for _, rs in groupby(seq, key):
        # to list or not to list
        yield Rows(rs)


def _build_keyfn(key):
    """
    Args:
        key (str or List[str]): column names
    Returns:
        FN[Row -> type]
    """
    # if the key is a function, just return it
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
    Args:
        columns (List[str])
    Returns:
        List[str]
    Example:
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
    Args:
        name (str): table name
        colnames (List[str])
    Returns:
        str
    """
    schema = ', '.join([col.lower() + ' ' + 'numeric' for col in colnames])
    return "create table if not exists %s (%s)" % (name.lower(), schema)


def _insert_statement(name, ncol):
    """insert into foo values (?, ?, ?, ...)

    Note:
        Column name is lower cased
    Args:
        name (str): table name
        ncol (int)
    Returns:
        str
    """
    qmarks = ', '.join(['?'] * ncol)
    return "insert into %s values (%s)" % (name.lower(), qmarks)


def _is_oneword(query):
    """
    Args:
        query (str)
    Returns:
        bool
    """
    return len(query.strip().split(' ')) == 1


def _select_statement(query, cols='*'):
    """If query is just one word, turn it to a select stmt
    or just leave it

    Args:
        query (str)
        cols (str or List[str])
    Returns:
        str
    """
    if _is_oneword(query):
        return "select %s from %s" % (', '.join(listify(cols)), query)
    return query


# The following 2 helpers are used in 'SQLPlus.save'
# just reel out sqlite3 rows and save sqlite3 rows
# not the rows from this script.
def _sqlite3_reel(cursor, table_name, column_names):
    q = _select_statement(table_name, column_names)
    yield from cursor.execute(q)


# srows: sqlite3 rows
def _sqlite3_save(cursor, srows, table_name, column_names):
    cursor.execute(_create_statement(table_name, column_names))
    istmt = _insert_statement(table_name, len(column_names))
    for r in srows:
        cursor.execute(istmt, r)


def _show(rows, n=30, cols=None, filename=None):
    """Printing to a screen or saving to a file

    Args:
        rows (Iter[Row])
        n (int): maximum number of lines to show
        cols (str or List[str]): columns to show
        filename (str): filename to save
    """
    # so that you can easily maintain code
    # Searching nrows is easier than searching n in editors
    nrows = n

    # then query is an iterator of rows, or a list of rows
    # of course it can be just a generator function of rows

    if cols:
        rows = _pick(cols, rows)

    row0, rows = peek_first(rows)

    colnames = row0.columns
    # Always use safer way
    seq_rvals = _safe_values(rows, colnames)

    # write to a file
    if filename:
        # practically infinite number
        nrows = nrows or sys.maxsize

        if not filename.endswith('.csv'):
            filename = filename + '.csv'

        with open(os.path.join(WORKSPACE, filename), 'w') as fout:
            w = csv.writer(fout)
            w.writerow(colnames)
            for rvals in islice(seq_rvals, nrows):
                w.writerow(rvals)

    # write to stdout
    else:
        # show practically all columns.
        with pd.option_context("display.max_rows", nrows), \
                pd.option_context("display.max_columns", 1000):
            # make use of pandas DataFrame displaying
            # islice 1 more rows than required
            # to see if there are more rows left
            seq_rvals_list = list(islice(seq_rvals, nrows + 1))
            print(pd.DataFrame(seq_rvals_list[:nrows],
                               columns=colnames))
            if len(seq_rvals_list) > nrows:
                print("...more rows...")


# sequence of row values to rows
def _build_rows(seq_rvals, colnames):
    for rvals in seq_rvals:
        r = Row()
        for col, val in zip(colnames, rvals):
            r[col] = val
        yield r
