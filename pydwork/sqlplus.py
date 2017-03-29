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
import io
import copy

from collections import Counter, OrderedDict
from contextlib import contextmanager
from itertools import groupby, islice, chain

import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import statistics as st
import warnings

from .util import isnum, istext, yyyymm, yyyymmdd, \
    listify, camel2snake, peek_first, parse_model, star, random_string

__all__ = ['dbopen', 'Row', 'Rows', 'set_workspace', 'Box']


WORKSPACE = ''

class Row:
    "Mutable version of sqlite3.Row"
    # Works for Python 3.6 and higher
    def __init__(self, **kwargs):
        super().__setattr__('_ordered_dict', OrderedDict())
        for k, v in kwargs.items():
            self._ordered_dict[k] = v

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
        content = ' | '.join(c + ': ' + str(v) for c, v in
                             zip(self.columns, self.values))
        return '[' + content + ']'

    # for pickling, very important
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    # TODO
    # hasattr doesn't for properly
    # You can't make it work by changing getters and setters
    # to an ordinary way. But it is slower


class Rows:
    """
    a shallow wrapper of a list of row instances """
    # Don't try to define __getattr__, __setattr__
    # List objects has a lot of useful attributes that can't be overwritten
    # Not the same situation as 'Row' class

    # Inheriting list can be problemetic
    # when you want to use this as a superclass
    # See 'where' method, you must return 'self' but it's not efficient
    # (at least afaik) if you inherit list
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    # __getitem__ enables you to iterate 'Rows'
    def __getitem__(self, cols):
        "cols: integer or list of strings or comma separated string"
        if isinstance(cols, int):
            return self.rows[cols]
        elif isinstance(cols, slice):
            # shallow copy for non-destructive slicing
            self = self.copy()
            self.rows = self.rows[cols]
            return self

        cols = listify(cols)
        if len(cols) == 1:
            col = cols[0]
            return [r[col] for r in self.rows]
        else:
            return [[r[c] for c in cols] for r in self.rows]

    def __setitem__(self, cols, vals):
        """vals can be just a list or a list of lists,
        demensions must match
        """
        if isinstance(cols, int) or isinstance(cols, slice):
            self.rows[cols] = vals
            return

        cols = listify(cols)
        ncols = len(cols)

        if not isinstance(vals, list):
            if ncols == 1:
                col = cols[0]
                for r in self.rows:
                    r[col] = vals
            else:
                for r in self.rows:
                    for c in cols:
                        r[c] = vals

        elif not isinstance(vals[0], list):
            if ncols != len(vals):
                raise ValueError('Number of values to assign inappropriate')
            for r in self.rows:
                for c, v in zip(cols, vals):
                    r[c] = v

        else:
            # validity check,
            if len(self.rows) != len(vals):
                raise ValueError('Number of values to assign inappropriate')

            # vals must be rectangular!
            if ncols > 1:
                for vs in vals:
                    if len(vs) != ncols:
                        raise ValueError('Invalid values to assign', vs)

            if ncols == 1:
                col = cols[0]
                for r, v in zip(self.rows, vals):
                    r[col] = v
            else:
                for r, vs in zip(self.rows, vals):
                    for c, v in zip(cols, vs):
                        r[c] = v

    def __delitem__(self, cols):
        if isinstance(cols, int) or isinstance(cols, slice):
            del self.rows[cols]
            return

        cols = listify(cols)
        ncols = len(cols)

        if ncols == 1:
            col = cols[0]
            for r in self.rows:
                del r[col]
        else:
            for r in self.rows:
                for c in cols:
                    del r[c]

    def __add__(self, other):
        self.rows = self.rows + other.rows
        return self

    def is_valid(self):
        cols = self.rows[0].columns
        for r in self.rows[1:]:
            assert r.columns == cols, str(r)
        return self

    def append(self, r):
        self.rows.append(r)
        return self

    def copy(self):
        "shallow copy"
        return copy.copy(self)

    def order(self, key, reverse=False):
        self.rows.sort(key=_build_keyfn(key), reverse=reverse)
        return self

    def where(self, pred):
        other = self.copy()
        pred = _build_keyfn(pred)
        other.rows = [r for r in self.rows if pred(r)]
        return other

    # num and text, I don't like the naming
    def num(self, cols):
        "another simplified filtering, numbers only"
        cols = listify(cols)
        return self.where(lambda r: all(isnum(r[col]) for col in cols))

    def wavg(self, col, wcol=None):
        if wcol:
            rs = self.num([col, wcol])
            total = sum(r[wcol] for r in rs)
            return sum(r[col] * r[wcol] / total for r in rs)
        else:
            return st.mean(r[col] for r in self.rows if isnum(r[col]))

    def text(self, cols):
        "another simplified filtering, texts(string) only"
        cols = listify(cols)
        return self.where(lambda r: all(istext(r[col]) for col in cols))

    def ols(self, model):
        # TODO: patsy raises some annoying warnings
        # Remove the following later
        warnings.filterwarnings("ignore")
        return sm.ols(formula=model, data=self.df()).fit()

    def reg(self, model):
        "we need some simple printing"
        result = self.ols(model)
        r1, r2 = Row(), Row()
        rows = Rows([r1, r2])
        for x, p in zip(result.params.iteritems(), result.pvalues):
            k, v = x
            r1[k] = star(v, p)
        for k, v in result.tvalues.iteritems():
            r2[k] = "[%s]" % (round(v, 2))
        rows['n, r2'] = ''
        r1.n = result.nobs
        r1.r2 = round(result.rsquared, 3)
        # You may need more
        other = self.copy()
        other.rows = rows.rows
        return other

    def truncate(self, col, limit=0.01):
        "Truncate extreme values, defalut 1 percent on both sides"
        xs = self[col]
        lower = np.percentile(xs, limit * 100)
        higher = np.percentile(xs, (1 - limit) * 100)
        return self.where(lambda r: r[col] >= lower and r[col] <= higher)

    def winsorize(self, col, limit=0.01):
        xs = self[col]
        lower = np.percentile(xs, limit * 100)
        higher = np.percentile(xs, (1 - limit) * 100)
        for r in self.rows:
            if r[col] > higher:
                r[col] = higher
            elif r[col] < lower:
                r[col] = lower
        return self

    def group(self, key):
        for rs in _gby(self.rows, key):
            other = self.copy()
            other.rows = rs.rows
            yield other

    def show(self, n=30, cols=None):
        if self == []:
            print(self.rows)
        else:
            _show(self.rows, n, cols)

    def desc(self, n=5, cols=None, percentile=None):
        if self.rows == []:
            print(self.rows)
        else:
            _describe(self.rows, n, cols, percentile)

    # write to csv file
    def csv(self, file=sys.stdout, cols=None):
        _csv(self.rows, file, cols)

    # Use this when you need to see what's inside
    # for example, when you want to see the distribution of data.
    def df(self, cols=None):
        if cols:
            cols = listify(cols)
            return pd.DataFrame([[r[col] for col in cols] for r in self.rows],
                                columns=cols)
        else:
            cols = self.rows[0].columns
            seq = _safe_values(self.rows, cols)
            return pd.DataFrame(list(seq), columns=cols)


class Box:
    """We need something very simple and flexible for displaying
    list of lists
    """
    def __init__(self, lines):
        self.lines = lines

    def csv(self, file=sys.stdout):
        _csv(self.lines, file, None)


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
    # It is unlikely that we need to worry about the security issues
    # but still there's no harm. So...
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

    def save(self, x, name=None, fn=None, overwrite=False, args=()):
        """create a table from an iterator.

        x (str or iter or GF[* -> Row])
        name (str): table name in DB
        fn: function that takes a row(all elements are strings)
            and returns a row, used for csv file transformation
        """
        # handle simple case first,
        # if x(string) starts with 'select' then you save it
        # (if no name is not given source table name is used for the new table)
        if isinstance(x, str) \
            and x.split()[0].lower() == 'select' \
            and (fn is None):

            return self._new(x, name, overwrite, args)

        name1, rows = _x2rows(x, self._cursor, args)
        name = name or name1
        if not name:
            raise ValueError('table name required')

        if (not overwrite) and (name in self.tables):
            return

        temp_name = 'table_' + random_string(10)

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

            _sqlite3_save(cursor, seq_values, temp_name, cols)
            _sqlite3_save(self._cursor, _sqlite3_reel(cursor, temp_name, cols),
                          temp_name, cols)

            self.run(f'drop table if exists { name }')
            self.run(f'alter table { temp_name } rename to { name }')
            # no need to commit and close the connection,
            # it's going to be erased anyway

        self.tables = self._list_tables()

    # Be careful so that you don't overwrite the file
    def show(self, x, n=30, cols=None, args=()):
        "Printing to a screen or saving to a file "
        _, rows = _x2rows(x, self._cursor, args)
        _show(rows, n, cols)

    def desc(self, query, n=5, cols=None, percentile=None, args=()):
        "Summary"
        _, rows = _x2rows(query, self._cursor, args)
        _describe(rows, n, cols, percentile)

    def csv(self, x, file=sys.stdout, cols=None, args=()):
        """
        """
        _, rows = _x2rows(x, self._cursor, args)
        _csv(rows, file, cols)

    def drop(self, tables):
        " drop table if exists "
        tables = listify(tables)
        for table in tables:
            # you can't use '?' for table name
            # '?' is for data insertion
            self.run('drop table if exists %s' % table)
        self.tables = self._list_tables()

    def rename(self, old, new):
        if old in self.tables:
            self.run(f'drop table if exists { new }')
            self.run(f'alter table { old } rename to { new }')

    def _list_tables(self):
        "List of table names in the database "
        query = self._cursor.execute("""
        select * from sqlite_master
        where type='table'
        """)
        # **.lower()
        tables = [row[1].lower() for row in query]
        return sorted(tables)

    def _new(self, query, name, overwrite, args):
        """Create new table from query
        """
        def get_name_from_query(query):
            query_list = query.lower().split()
            idx = query_list.index('from')
            return query_list[idx + 1]

        temp_name = 'table_' + random_string(10)
        name = name if name else get_name_from_query(query)

        if (not overwrite) and (name in self.tables):
            return

        self.run(f"create table if not exists { temp_name } as { query }",
                 args=args)
        self.run(f'drop table if exists { name }')
        self.run(f"alter table { temp_name } rename to { name }")


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
    WORKSPACE = path if os.path.isabs(path) else \
        os.path.join(os.getcwd(), path)


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

        # reader = csv.reader(fin)
        # NULL byte error handling
        reader = csv.reader(x.replace('\0', '') for x in fin)
        for line_no, line in enumerate(reader, 2):
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


def _write_all(lines, file):
    "Write all to csv"
    w = csv.writer(file)
    for line in lines:
        w.writerow(line)


def _csv(rows, file, cols):
    if cols:
        rows = _pick(cols, rows)

    row0, rows1 = peek_first(rows)
    if isinstance(row0, Row):
        seq_values = chain([row0.columns], _safe_values(rows1, row0.columns))
    else:
        seq_values = rows1

    if file == sys.stdout:
        _write_all(seq_values, file)
    elif isinstance(file, str):
        try:
            fout = open(os.path.join(WORKSPACE, file), 'w')
            _write_all(seq_values, fout)
        finally:
            fout.close()
    elif isinstance(file, io.TextIOBase):
        try:
            _write_all(seq_values, file)
        finally:
            file.close()
    else:
        raise ValueError('Invalid file', file)


def _show(rows, n, cols):
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

    with pd.option_context("display.max_rows", nrows), \
            pd.option_context("display.max_columns", 1000):
        # make use of pandas DataFrame displaying
        # islice 1 more rows than required
        # to see if there are more rows left
        list_values = list(islice(seq_values, nrows + 1))
        print(pd.DataFrame(list_values[:nrows], columns=cols))
        if len(list_values) > nrows:
            print("...more rows...")


#  temporary, you need to fix it later
def _describe(rows, n, cols, percentile):
    def fill(xs, cols):
        d = {}
        for a, b in zip(xs.index, xs):
            d[a] = b

        result = []
        for c in cols:
            if c not in d:
                result.append(float('nan'))
            else:
                result.append(d[c])
        return result

    rows1 = Rows(rows)
    percentile = percentile if percentile else \
        [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    df = rows1.df(cols)
    desc = df.describe(percentile, include='all')
    desc.loc['skewness'] = fill(df.skew(), desc.columns)
    desc.loc['kurtosis'] = fill(df.kurtosis(), desc.columns)

    r = Row()
    for c in rows1[0].columns:
        r[c] = '***'

    rows1.rows = [r] + rows1.rows
    print()
    print(pd.concat([desc, rows1[:n + 1].df()]))
    print()

    corr1 = df.corr()
    corr2 = df.corr('spearman')
    columns = list(corr1.columns.values)

    lcorr1 = corr1.values.tolist()
    lcorr2 = corr2.values.tolist()
    for i in range(len(columns)):
        for j in range(i):
            lcorr2[i][j] = lcorr1[i][j]
    for i in range(len(columns)):
        lcorr2[i][i] = ''
    result = []
    for c, ls in zip(columns, lcorr2):
        result.append([c] + ls)

    print(pd.DataFrame(result, columns=['Pearson\\Spearman'] + columns).
          to_string(index=False))


# sequence row values to rows
def _build_rows(seq_values, cols):
    "build rows from an iterator of values"
    for vals in seq_values:
        r = Row()
        for col, val in zip(cols, vals):
            r[col] = val
        yield r
