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

from collections import Counter
from contextlib import contextmanager
from functools import wraps
from itertools import groupby, islice

import pandas as pd

from .util import isnum, istext, yyyymm, listify, camel2snake, peek_first


__all__ = ['dbopen', 'Row', 'Rows', 'reel', 'adjoin', 'disjoin',
           'todf', 'torows',
           'set_workspace', 'get_workspace']


WORKSPACE = ''


class Row:
    """
    Basically the same as sqlite3.Row
    it's just that using sqlite3.Row is a bit clunkier.
    r['col'] = 34
    I prefer r.col = 34

    And it's better to keep the order of columns
    """

    def __init__(self):
        super().__setattr__('_columns', [])
        super().__setattr__('_values', [])

    @property
    def columns(self):
        "List[str]: column names"
        return self._columns

    @property
    def values(self):
        "List[type]"
        return self._values

    def __setattr__(self, name, value):
        if name not in self.__dict__:
            self.columns.append(name)
            self.values.append(value)
        else:
            # Updating the attribute value of a row
            idx = self.columns.index(name)
            self.values[idx] = value
        super().__setattr__(name, value)

    def __delattr__(self, name):
        idx = self.columns.index(name)
        del self.columns[idx]
        del self.values[idx]
        super().__delattr__(name)

    def __str__(self):
        return str(list(zip(self.columns, self.values)))


class Rows(list):
    """
    a shallow wrapper of a list of row instances
    """

    def add(self, colname, xs):
        if len(self) != len(xs):
            raise valueerror('length of list not matched')

        for r, x in zip(self, xs):
            setattr(r, colname, x)
        return self

    def col(self, colname):
        xs = []
        for r in self:
            xs.append(getattr(r, colname))
        return xs

    def cols(self, colnames):
        "returns a list of lists"
        colnames = listify(colnames)
        xs = []
        for r in self:
            xs.append([getattr(r, col) for col in colnames])
        return xs

    def order(self, key, reverse=0):
        key = _build_keyfn(key)
        self.sort(key=key, reverse=reverse)
        return self

    def filter(self, pred):
        pred = _build_keyfn(pred)
        return Rows([r for r in self if pred(r)])

    def fromto(self, col, beg, end):
        "simplified fitering, inclusive"
        result = []
        for r in self:
            val = getattr(r, col)
            if beg <= val and val <= end:
                result.append(r)
        return Rows(result)

    def ge(self, col, beg):
        "greater than or equal to"
        result = []
        for r in self:
            val = getattr(r, col)
            if beg <= val:
                result.append(r)
        return Rows(result)

    def le(self, col, end):
        "less than or equal to"
        result = []
        for r in self:
            val = getattr(r, col)
            if val <= end:
                result.append(r)
        return Rows(result)

    def num(self, cols):
        "another simplified filtering, numbers only"
        cols = listify(cols)
        result = []
        for r in self:
            if all(isnum(getattr(r, col)) for col in cols):
                result.append(r)
        return Rows(result)

    def contains(self, col, vals):
        vals = listify(vals)
        result = []
        for r in self:
            if getattr(r, col) in vals:
                result.append(r)
        return Rows(result)

    def group(self, key):
        # it is illogical to return an instance of Rows
        result = []
        for rs in gby(self, key):
            result.append(rs)
        return result

    def show(self, n=30, cols=None, filename=None, overwrite=True):
        if self == []:
            print(self)
        else:
            _show(self, n=n, cols=cols, filename=filename, overwrite=overwrite)

    # Simpler version of show (when you write it to a file)
    def write(self, filename):
        _show(self, n=None, filename=filename, overwrite=True)

    def df(self):
        return todf(self)


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

    # args can be a list, a tuple or a dictionary
    def run(self, query, args=()):
        """Simply executes sql statement and update tables attribute

        Args:
            query (str): SQL query string
            args (List[any] or Tuple[any]): args for SQL query
        """
        self._cursor.execute(query, args)
        self.tables = self._list_tables()

    def reel(self, query, args=(), group=False):
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
            yield from gby(_build_rows(qrows, columns), group)
        else:
            yield from _build_rows(qrows, columns)

    def save(self, seq, name=None, args=(), n=None, overwrite=False):
        """create a table from an iterator.

        Note:<%=  %>
            if seq is a generator function and 'name' is not given,
            the function name is going to be the table name.

        Args:
            seq (iter or GF[* -> Row])
            name (str): table name in DB
            args (List[type]): args for seq (GF)
            n (int): number of rows to save
        """
        # single letter variable is hard to find
        nrows = n

        # if 'seq' is a generator function, it is executed to make an iterator
        if hasattr(seq, '__call__'):
            name = name or seq.__name__
            seq = seq(*args)

        if name is None:
            raise ValueError('table name required')

        if overwrite:
            self.run('drop table if exists %s' % name)

        if name in self.tables:
            return

        if nrows:
            seq = islice(seq, nrows)

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
            _sqlite3_save(cursor, (r.values for r in seq), name, colnames)
            _sqlite3_save(self._cursor, _sqlite3_reel(cursor, name, colnames),
                          name, colnames)
            # no need to commit and close the connection,
            # it's going to be erased anyway

        self.tables = self._list_tables()

    # Be careful so that you don't overwrite the file
    def show(self, query, args=(), n=30, cols=None,
             filename=None, overwrite=True):
        """Printing to a screen or saving to a file

        Args:
            query (str or Iter[Row] or GF)
            args (List[type] or Tuple[type]): args for query (GF)
            n (int): maximum number of lines to show
            cols (str or List[str]): columns to show
            filename (str): filename to save
            overwrite (bool): if true overwrite a file
        """
        # so that you can easily maintain code
        # Searching nrows is easier than searching n in editors
        nrows = n

        if isinstance(query, str):
            seq_rvals = self._cursor.execute(_select_statement(query), args)
            colnames = [c[0] for c in seq_rvals.description]
            rows = _build_rows(seq_rvals, colnames)

        # then query is an iterator of rows, or a list of rows
        # of course it can be just a generator function of rows
        else:
            rows = query
            if hasattr(rows, '__call__'):
                rows = rows(*args)

        _show(rows, n=nrows, cols=cols, filename=filename, overwrite=overwrite)

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
        self.show(query, filename=filename, args=args, n=None, overwrite=True)

    def _list_tables(self):
        """List of table names in the database

        Returns:
            List[str]
        """
        query = self._cursor.execute("""
        select * from sqlite_master
        where type='table'
        """)
        tables = [row[1].lower() for row in query]
        return sorted(tables)

    def summarize(self, n=1000, overwrite=True):
        """
        Args:
            n (int)
            overwrite (bool)
        """
        summary_dir = os.path.join(WORKSPACE, 'summary')
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

        for table in self.tables:
            filename = os.path.join(summary_dir, table + '.csv')
            self.show(table, n=n, filename=filename, overwrite=overwrite)

    def drop(self, tables):
        """
        drop table if exists

        Args:
            tables (str or List[str])
        """
        # you can't use '?' for table name
        # '?' is for data insertion
        tables = listify(tables)
        summary_dir = os.path.join(WORKSPACE, 'summary')
        for table in tables:
            self.run('drop table if exists %s' % table)
            filename = os.path.join(summary_dir, table + '.csv')
            if os.path.isfile(filename):
                # remove summary file as well if exists
                os.remove(filename)
        self.tables = self._list_tables()


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


def gby(seq, key):
    """Group the iterator by a key

    Args
        seq (iter)
        key (FN[Row -> type] or List[str] or str): if [], group them all
    Yields:
        List[Row]
    """
    key = _build_keyfn(key)
    for _, rows in groupby(seq, key):
        # to list or not to list
        yield list(rows)


def todf(rows):
    """
    Args:
        List[Row]
    Returns:
        pd.DataFrame
    """
    colnames = rows[0].columns
    d = {}
    for col in zip(colnames, *(r.values for r in rows)):
        d[col[0]] = col[1:]
    return pd.DataFrame(d)


# This is not an ordinary function!!
# So, x != torows(todf(x))
# efficiency issue
# Most of the time, in fact almost always,
# you will use this function to use with yield from
# so there's no point in returning a list of rows
def torows(df):
    """
    Args:
        df (pd.DataFrame)
    Yields:
        Row
    """
    colnames = df.columns.values
    for vals in df.values.tolist():
        r = Row()
        for c, v in zip(colnames, vals):
            setattr(r, c, v)
        yield r


def pick(cols, seq):
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
            setattr(r1, c, getattr(r, c))
        yield r1



# consider changing the name to reel_csv
# EVERY COLUMN IS A STRING!!!
def reel(csv_file, header=None, group=False):
    """Loads well-formed csv file, 1 header line and the rest is data

    Args:
        csv_file (str)
        header (str)
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
        header = header or first_line
        columns = _gen_valid_column_names(listify(header))
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
                    setattr(row1, col, val)
                yield row1

        if group:
            yield from gby(rows(), group)
        else:
            yield from rows()


def adjoin(colnames):
    """Decorator to ensure that the rows to have the columns for sure

    Args:
        colnames (str or List[str])
    Returns:
        FN
    """
    colnames = listify(colnames)

    def dec(gen):
        "real decorator"
        @wraps(gen)
        def wrapper(*args, **kwargs):
            "if a column doesn't exist, append it"
            for row in gen(*args, **kwargs):
                for col in colnames:
                    try:
                        # rearrange the order
                        val = getattr(row, col)
                        delattr(row, col)
                        setattr(row, col, val)
                    except:
                        setattr(row, col, '')
                yield row
        return wrapper
    return dec


def disjoin(colnames):
    """Decorator to ensure that the rows NOT to have the columns for sure.

    Args:
        colnames (str or List[str])
    Returns:
        FN
    """
    colnames = listify(colnames)

    def dec(gen):
        "real decorator"
        @wraps(gen)
        def wrapper(*args, **kwargs):
            "Delete a column"
            for row in gen(*args, **kwargs):
                for col in colnames:
                    # whatever it is, just delete it
                    try:
                        delattr(row, col)
                    except:
                        pass
                yield row
        return wrapper
    return dec


def set_workspace(dir):
    """
    Args:
        dir (str)
    """
    global WORKSPACE

    WORKSPACE = dir if os.path.isabs(dir) else os.path.join(os.getcwd(), dir)


def get_workspace():
    """
    Returns:
        str
    """
    return WORKSPACE


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
        return lambda r: getattr(r, colnames[0])
    else:
        return lambda r: [getattr(r, colname) for colname in colnames]


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
    SQLITE_KEYWORDS = {
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

    DEFAULT_COLUMN_NAME = 'col'
    temp_columns = []
    for col in columns:
        # save only alphanumeric and underscore
        # and remove all the others
        newcol = camel2snake(re.sub(r'[^\w]+', '', col))
        if newcol == '':
            newcol = DEFAULT_COLUMN_NAME
        elif not newcol[0].isalpha() or newcol.upper() in SQLITE_KEYWORDS:
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


def _show(rows, n=30, cols=None, filename=None, overwrite=True):
    """Printing to a screen or saving to a file

    Args:
        rows (Iter[Row])
        n (int): maximum number of lines to show
        cols (str or List[str]): columns to show
        filename (str): filename to save
        overwrite (bool): if true overwrite a file
    """
    # so that you can easily maintain code
    # Searching nrows is easier than searching n in editors
    nrows = n

    # then query is an iterator of rows, or a list of rows
    # of course it can be just a generator function of rows

    if cols:
        rows = pick(cols, rows)

    row0, rows = peek_first(rows)

    colnames = row0.columns
    seq_rvals = (r.values for r in rows)

    # write to a file
    if filename:
        # practically infinite number
        nrows = nrows or sys.maxsize

        if not filename.endswith('.csv'):
            filename = filename + '.csv'

        if os.path.isfile(os.path.join(WORKSPACE, filename)) \
           and not overwrite:
            return

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
            setattr(r, col, val)
        yield r
