"""
'sqlite3' based SQL utils

Say if you want to group rows and do some complex jobs,
afaik, you can't simply do it with SQL.

And if the data size is so humongous that your system can't load
all up in the memory, you can't easily use pandas either.

This program is not a sophisticated, full-fledged automation system.
I'd rather say this is just a mental framework
as for data wrangling.

It was a little clunkier experience to learn pandas.
And I find basic Python syntax is good enough for most of data wrangling jobs.
And also, SQL is a language as wonderful as Python.
If you already know SQL and python, all you need to think about is
how you combine them both.

This program does it.

What you need to know is in unit test code at test/*
"""

import os
import csv
import fileinput
import re
import sqlite3
import tempfile
# memory check
import psutil
import sys
import heapq


from collections import Counter
from contextlib import contextmanager
from functools import wraps
from itertools import chain, groupby, islice, dropwhile

from bs4 import BeautifulSoup

import pandas as pd

__all__ = ['dbopen', 'Row', 'gby', 'gflat', 'reel',
           'read_html_table', 'ljoin1', 'ljoin', 'show', 'drop',
           'add_header', 'del_header', 'adjoin', 'disjoin', 'select', 'todf',
           'sortl', 'set_workspace']

# Some of the sqlite keywords are not allowed for column names
# http://www.sqlite.org/sessions/lang_keywords.html
SQLITE_KEYWORDS = [
    "ABORT", "ACTION", "ADD", "AFTER", "ALL", "ALTER", "ANALYZE", "AND",
    "AS", "ASC", "ATTACH", "AUTOINCREMENT", "BEFORE", "BEGIN", "BETWEEN",
    "BY", "CASCADE", "CASE", "CAST", "CHECK", "COLLATE", "COLUMN", "COMMIT",
    "CONFLICT", "CONSTRAINT", "CREATE", "CROSS", "CURRENT_DATE",
    "CURRENT_TIME", "CURRENT_TIMESTAMP", "DATABASE", "DEFAULT", "DEFERRABLE",
    "DEFERRED", "DELETE", "DESC", "DETACH", "DISTINCT", "DROP", "EACH", "ELSE",
    "END", "ESCAPE", "EXCEPT", "EXCLUSIVE", "EXISTS", "EXPLAIN", "FAIL",
    "FOR", "FOREIGN", "FROM", "FULL", "GLOB", "GROUP", "HAVING", "IF",
    "IGNORE", "IMMEDIATE", "IN", "INDEX", "INDEXED", "INITIALLY", "INNER",
    "INSERT", "INSTEAD", "INTERSECT", "INTO", "IS", "ISNULL", "JOIN", "KEY",
    "LEFT", "LIKE", "LIMIT", "MATCH", "NATURAL",
    # no is ok somehow
    # no idea why
    # "NO",
    "NOT",
    "NOTNULL", "NULL", "OF", "OFFSET", "ON", "OR", "ORDER", "OUTER", "PLAN",
    "PRAGMA", "PRIMARY", "QUERY", "RAISE", "REFERENCES", "REGEXP", "REINDEX",
    "RENAME", "REPLACE", "RESTRICT", "RIGHT", "ROLLBACK", "ROW", "SAVEPOINT",
    "SELECT", "SET", "TABLE", "TEMP", "TEMPORARY", "THEN", "TO", "TRANSACTION",
    "TRIGGER", "UNION", "UNIQUE", "UPDATE", "USING", "VACUUM", "VALUES",
    "VIEW", "VIRTUAL", "WHEN", "WHERE"
]


WORKSPACE = ''


# Most of the time you work with SQL rows.
# You append, delete, replace columns.
# Or you combine rows with similar properties and think of them as a bunch.
# And then later flatten'em if you want.
class Row:
    """SQL row

    Pretty much nothing, but essential part of this program.
    """

    def __init__(self):
        """Example:

        r = Row()
        r.x = 10; r.y = 20
        del r.x
        """
        # To preserve orders
        super().__setattr__('columns', [])

    def get_values(self, columns):
        """Returns a list of values

        Args
            columns: list of column name strings
        """
        return [getattr(self, c) for c in columns]

    def __getattr__(self, name):
        "if the attribute doesn't exist just return ''"
        try:
            val = super().__getattr__(name)
        except AttributeError:
            return ''
        return val

    def __setattr__(self, name, value):
        if name == 'columns':
            raise AttributeError("'columns' not allowed")

        if name not in self.columns:
            self.columns.append(name)
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name == 'columns':
            raise AttributeError("'columns' not allowed")
        try:
            del self.columns[self.columns.index(name)]
        except:
            raise AttributeError("Does not exist", name)
        super().__delattr__(name)

    def __str__(self):
        return str(list(zip(self.columns, self.get_values(self.columns))))


class SQLPlus:
    """SQLPlus object works like a sql cursor.
    """

    def __init__(self, dbfile):
        # self._dbfile = os.path.join(WORKSPACE, dbfile)
        self._dbfile = dbfile if dbfile == ':memory:' \
            else os.path.join(WORKSPACE, dbfile)
        self.conn = sqlite3.connect(self._dbfile)
        self._cursor = self.conn.cursor()
        self.tables = self.list_tables()

    # args can be a list, a tuple or a dictionary
    # <- fix it
    def run(self, query, args=()):
        """Simply executes sql statement and update tables attribute
        """
        self._cursor.execute(query, args)
        # update tables
        self.tables = self.list_tables()

    def reel(self, query, args=()):
        """Generates a sequence of rows from a query.

        Query can be a select statement or table name.
        """
        query = _select_statement(query)
        if query.strip().partition(' ')[0].upper() != "SELECT":
            raise ValueError("use 'run' for ", query)
        qrows = self._cursor.execute(query, args)
        columns = [c[0] for c in qrows.description]
        for qrow in qrows:
            row = Row()
            for col, val in zip(columns, qrow):
                setattr(row, col, val)
            yield row

    def save(self, seq, name=None, args=(), n=None):
        """create a table from an iterator.

        seq is an iterator or a generator function.
        if seq is a generator function and 'name' is not given,
        the function name is going to be the table name.

        'args' are going to be passed as arguments for the generator function
        """

        # for maintenance
        nrow = n

        # if 'it' is a generator function, it is executed to make an iterator
        if hasattr(seq, '__call__'):
            name = name or seq.__name__
            seq = seq(*args)

        # tests empty sequence
        try:
            row0, seq = _peek_first(seq)
        except:
            print('\nEmpty Sequence, Nothing to Save')
            return

        # implicitly gflat
        seq = gflat(seq)

        if nrow:
            seq = islice(seq, nrow)

        if name in self.tables:
            return
        if name is None:
            raise ValueError('name should be passed')

        colnames = row0.columns

        # You can't save the iterator directly because
        # once you execute a table creation query,
        # then the query becomes the most recent query,
        # not the query for the iterator.
        # Which means, if you iterate over the iterator,
        # it iterates over the table creation query.

        # You can see the example at test/sqlplus_test.py
        # 'test_run_over_run'

        # So you save the iterator up in a temporary file
        # and then save the file to a database.
        # In the process column types are checked.

        # This is the root of all evil
        with tempfile.TemporaryFile() as fport:
            # Write the iterator in a temporary file
            # encode it as binary.
            _save_rows_to_tempfile(fport, seq, colnames)

            # create table
            istmt = _insert_statement(name, len(colnames))
            self._cursor.execute(_create_statement(name, colnames))

            # Now insertion to a DB
            fport.seek(0)
            # read out the first line, header column
            fport.readline()

            for line in fport:
                # line[:-1] because last index indicates '\n'
                try:
                    line_vals = line[:-1].decode().split(',')
                    self._cursor.execute(istmt, line_vals)
                except:
                    pass
                    # raise ValueError("Invalid line to save", line_vals)
        self.tables.append(name)

    # Be careful so that you don't overwrite the file
    def show(self, query, args=(), filename=None, desc=None, n=30, cols=None):
        """Printing to a screen or saving to a file

        'query' can be either a SQL query string or an iterable.
        'n' is a maximun number of rows to show up,
        """
        # so that you can easily maintain code
        # Searching nrows is easier than searching n in editors
        nrows = n

        if isinstance(query, str):
            seq_rvals = self._cursor.execute(
                _select_statement(query, cols), args)
            colnames = [c[0] for c in seq_rvals.description]

        # then query is an iterator of rows, or a list of rows
        # of course it can be just a generator function of rows
        else:
            rows = query
            if hasattr(rows, '__call__'):
                rows = rows(*args)

            if cols:
                rows = select(rows, cols=cols)
            try:
                row0, rows = _peek_first(rows)
            except:
                print('\nEmpty Sequence')
                return

            colnames = row0.columns
            # implicit gflat
            seq_rvals = (r.get_values(colnames) for r in gflat(rows))

        if filename:
            # ignore n

            if not filename.endswith('.csv'):
                filename = filename + '.csv'
            # write file description
            if not os.path.isfile(os.path.join(WORKSPACE, filename)):
                if desc:
                    with open(os.path.join(filename[:-4] + '.desc')) as f:
                        f.write(desc)

                with open(os.path.join(WORKSPACE, filename), 'w') as fout:
                    fout.write(','.join(colnames) + '\n')
                    for rvals in seq_rvals:
                        fout.write(','.join([str(val) for val in rvals]) +
                                   '\n')
        else:
            # show practically all rows, columns.
            with pd.option_context("display.max_rows", nrows), \
                    pd.option_context("display.max_columns", 1000):
                # make use of pandas DataFrame displaying
                seq_rvals_list = list(islice(seq_rvals, nrows + 1))
                print(pd.DataFrame(seq_rvals_list[:nrows],
                                   columns=colnames))
                if len(seq_rvals_list) > nrows:
                    print("...more rows...")

    def list_tables(self):
        """List of table names in the database
        """
        query = self._cursor.execute("""
        select * from sqlite_master
        where type='table'
        """)
        tables = [row[1] for row in query]
        return sorted(tables)


@contextmanager
def dbopen(dbfile):
    """Connects to SQL database(sqlite)
    """
    splus = SQLPlus(dbfile)
    try:
        yield splus
    finally:
        splus.conn.commit()
        splus.conn.close()


# 'grouped row' refers to a Row object
# with all-list properties
def gby(seq, key, bind=True):
    """Group the iterator by columns

    Depends heavily on 'groupby' from itertools

    Args
        seq: an iterator
        group: Either a function, or a comma(space) separated string,
               or a list(tuple) of strings
               or [] to group them all
        bind: True, for a grouped rows
              False, for a list of rows
    """
    def _grouped_row(rows, columns):
        """Returns a grouped row, from a list of simple rows
        """
        g_row = Row()
        for col in columns:
            setattr(g_row, col, [])
        for row1 in rows:
            for col in columns:
                getattr(g_row, col).append(getattr(row1, col))
        return g_row

    if not hasattr(key, '__call__'):
        key = _build_keyfn(key)
    g_seq = groupby(seq, key)

    if bind:
        first_group = list(next(g_seq)[1])
        colnames = first_group[0].columns

        yield _grouped_row(first_group, colnames)
        for _, rows in g_seq:
            yield _grouped_row(rows, colnames)
    else:
        for _, rows in g_seq:
            yield list(rows)


def todf(g_row):
    "A grouped row to a data from from pandas"
    return pd.DataFrame({col: getattr(g_row, col) for col in g_row.columns})


def gflat(seq):
    """Turn an iterator of grouped rows into an iterator of simple rows.
    """
    def tolist(val):
        "if val is not a list then make it a list"
        if isinstance(val, list) or isinstance(val, pd.core.series.Series):
            return list(val)
        else:
            return [val]

    row0, seq = _peek_first(seq)

    colnames = list(row0.columns)

    for row in seq:
        for vals in zip(*(tolist(getattr(row, col)) for col in colnames)):
            new_row = Row()
            for col, val in zip(colnames, vals):
                setattr(new_row, col, val)
            yield new_row


def select(seq, cols=None, where=None, order=None, mem=0.3):
    def colsfn(cols):
        if not hasattr(cols, '__call__'):
            colnames = _listify(cols)

            def fn(r):
                newr = Row()
                for c in colnames:
                    setattr(newr, c, getattr(r, c))
                return newr
            return fn
        else:
            return cols

    def id(x): return x

    if cols:
        cols = colsfn(cols)
    else:
        cols = id

    if where:
        seq = (cols(r) for r in seq if where(r))

    if order:
        yield from sortl((cols(r) for r in seq), key=order, mem=mem)
    else:
        yield from (cols(r) for r in seq)


# Some files don't have a header
def add_header(filename, header):
    """Adds a header line to an existing file.
    """
    for line in fileinput.input(os.path.join(WORKSPACE, filename),
                                inplace=True):
        if fileinput.isfirstline():
            print(header)
        print(line, end='')


def del_header(filename, num=1):
    """Delete n lines from a file
    """
    for line_number, line in enumerate(
            fileinput.input(os.path.join(WORKSPACE, filename), inplace=True)):
        if line_number >= num:
            print(line, end='')


def convtype(val):
    "convert type if possible"
    try:
        return int(val)
    except:
        try:
            return float(val)
        except:
            return val


def reel(csv_file, header=None, line_fix=(lambda x: x)):
    """Loads well-formed csv file, 1 header line and the rest is data

    returns an iterator
    All columns are string, no matter what.
    it's intentional. Types are guessed once it is saved in DB
    """

    with open(os.path.join(WORKSPACE, csv_file)) as fin:
        first_line = fin.readline()[:-1]
        header = header or first_line
        columns = _gen_valid_column_names(_listify(header))
        ncol = len(columns)
        for line in csv.reader(fin):
            if len(line) != ncol:
                if _is_empty_line(line):
                    continue
                line = line_fix(line)
                # if it's still not valid
                if len(line) != ncol:
                    raise ValueError("column number mismatch", columns, line)
            row1 = Row()
            for col, val in zip(columns, line):
                setattr(row1, col, convtype(val))
            yield row1


# Inflexible, experimental
def read_html_table(html_file, css_selector='table'):
    """Read simple well formed table
    """
    with open(os.path.join(WORKSPACE, html_file)) as fin:
        soup = BeautifulSoup(fin, 'html.parser')
        trs = soup.select(css_selector)[0].select('tr')
        colnames = _gen_valid_column_names(
            [x.text for x in trs[0].select('th')])
        for tr in trs[1:]:
            r = Row()
            vals = [x.text for x in tr.select('td')]
            for col, val in zip(colnames, vals):
                setattr(r, col, convtype(val))
            yield r


def show(seq, cols=None, where=None,
         order=None, n=30, filename=None, desc=None):
    with dbopen(':memory:') as c:
        c.show(select(seq, where=where, order=order),
               n=n, cols=cols, filename=filename, desc=desc)


def drop(filename):
    if not filename.endswith('.csv'):
        filename = filename + '.csv'
    os.remove(os.path.join(WORKSPACE, filename))
    descfile = os.path.join(filename[:-4] + '.desc')
    if os.path.isfile(descfile):
        os.remove(descfile)


def adjoin(colnames):
    """Decorator to ensure that the rows to have the columns for sure
    """
    def dec(gen):
        "real decorator"
        @wraps(gen)
        def wrapper(*args, **kwargs):
            "if a column doesn't exist, append it"
            for row in gen(*args, **kwargs):
                # row must be a Row instance,
                # Do not use this for dataframes, it's really not necessary
                assert isinstance(row, Row)
                for col in _listify(colnames):
                    try:
                        # rearrange the order
                        val = getattr(row, col)
                        delattr(row, col)
                        setattr(row, col, val)
                    except AttributeError:
                        setattr(row, col, '')
                yield row
        return wrapper
    return dec


def disjoin(colnames):
    """Decorator to ensure that the rows are missing
    """
    def dec(gen):
        "real decorator"
        @wraps(gen)
        def wrapper(*args, **kwargs):
            "Delete a column"
            for row in gen(*args, **kwargs):
                assert isinstance(row, Row)
                # row must be a Row instance,
                # Do not use this for dataframes, it's really not necessary
                for col in _listify(colnames):
                    # whatever it is, just delete it
                    try:
                        delattr(row, col)
                    except:
                        pass
                yield row
        return wrapper
    return dec


# TODO: decide whether n or mem(memory usage ratio is better
def sortl(seq, key=None, reverse=False, mem=0.3):
    """
    Sort large sequence, so large that the system memmory can't hold it
    n(int): chunk size to sort
    """
    try:
        row0, seq = _peek_first(seq)
    except:
        # stop earlier for empty sequence
        # TODO: There must be a more elegant way.
        yield from []
        return

    rowsize = sys.getsizeof(row0)
    available_memory = psutil.virtual_memory().available
    n = int((available_memory / rowsize) * mem)

    if key and not hasattr(key, '__call__'):
        key = _build_keyfn(key)

    colnames = row0.columns

    iters = []
    fs = []

    try:
        while True:
            rs = sorted(islice(seq, n), key=key, reverse=reverse)
            if not rs:
                break
            f = tempfile.TemporaryFile()
            _save_rows_to_tempfile(f, rs, colnames)
            f.seek(0)
            iters.append(_load_rows_from_tempfile(f))
            fs.append(f)
        for r in heapq.merge(*iters, key=key, reverse=reverse):
            yield r
        # if you don't close the files,
        # python complains
    finally:
        for f in fs:
            f.close()


def ljoin(first, rest, key, mem=0.3):
    """
    """
    for seq in rest:
        first = ljoin1(first, seq, key, mem=mem)
    yield from first


# TODO: super ugly, clean up
# You may consider using npc library
# I'd rather not now
def ljoin1(first, second, key, mem=0.3):
    """
    """
    def merge(r0, r1):
        # Maybe I am too timid not to modify r0
        r = Row()
        for c in r1.columns:
            setattr(r, c, getattr(r1, c))
        for c in r0.columns:
            setattr(r, c, getattr(r0, c))
        return r

    def merge_null(r0, columns):
        r = Row()
        for c in columns:
            setattr(r, c, None)
        for c in r0.columns:
            setattr(r, c, getattr(r0, c))
        return r

    if not hasattr(key, '__call__'):
        key = _build_keyfn(key)

    second0, second = _peek_first(second)
    second_columns = second0.columns

    second = sortl(second, key=key, mem=mem)
    keyval0 = None
    rs0 = None
    for r0 in sortl(first, key=key, mem=mem):
        keyval1 = key(r0)
        if keyval0 == keyval1:
            # same key value again
            yield from rs0
        else:
            keyval0 = keyval1

            second = dropwhile(lambda r: key(r) < keyval1, second)
            xs, extra = _takewhile(lambda r: key(r) == keyval1, second)
            # put extra back to second(rows)
            second = chain([extra], second)

            if xs:
                rs0 = [merge(r0, r1) for r1 in xs]
                yield from rs0
            else:
                yield merge_null(r0, second_columns)


# Should I just export WORKSPACE variable directly?
def set_workspace(dir):
    global WORKSPACE
    WORKSPACE = dir


# itertools.dropwhile is just ok to use right away
# but itertools.takewhile consumes one more element
# so you need to put it back later
def _takewhile(predicate, iterable):
    """A bit tricky
    """
    xs = []
    for x in iterable:
        if predicate(x):
            xs.append(x)
        else:
            break
    # you have consumed one more element that doens't satisfy predicate
    # so you return it together with the result you wanted
    return xs, x


def _build_keyfn(key):
    """
    If key is not a function, but a string or a list of strings
    that represents columns, turn it into a function
    """
    colnames = _listify(key)
    if len(colnames) == 1:
        return lambda r: getattr(r, colnames[0])
    else:
        return lambda r: [getattr(r, colname) for colname in colnames]


def _save_rows_to_tempfile(f, rs, colnames):
    def transform_value(val):
        """If val contains a comma or newline it causes problems
        So just remove them.
        There might be some other safer methods but I don't think
        newlines or commas are going to affect any data analysis.
        """
        return str(val).replace(',', ' ').replace('\n', ' ')

    f.write((','.join(colnames) + '\n').encode())
    # implicitly flatten
    for r in rs:
        vals = [transform_value(v) for v in r.get_values(colnames)]
        f.write((','.join(vals) + '\n').encode())


def _load_rows_from_tempfile(f):
    def splitit(line):
        # line[:-1] because last index indicates '\n'
        return line[:-1].decode().split(',')
    columns = splitit(f.readline())
    n = len(columns)
    for line in f:
        r = Row()
        vals = splitit(line)
        if len(vals) != n:
            # TODO: Error or pass that is the question
            # there are too many wierd lines for my job
            continue
        for name, val in zip(columns, vals):
            setattr(r, name, convtype(val))
        yield r


# TODO: unnecessarily complex
def _gen_valid_column_names(columns):
    """generate valid column names automatically

    ['a', '_b', 'a', 'a1"*c', 'a1c'] => ['a0', 'a_b', 'a1', 'a1c0', 'a1c1']
    """
    temp_columns = []
    for col in columns:
        # save only alphanumeric and underscore
        # and remove all the others
        newcol = re.sub(r'[^\w]+', '', col)
        if newcol == '':
            newcol = 'temp'
        elif not newcol[0].isalpha():
            newcol = 'a_' + newcol
        elif newcol.upper() in SQLITE_KEYWORDS:
            newcol = 'a_' + newcol
        temp_columns.append(newcol)

    # no duplicates
    if len(temp_columns) == len(set(temp_columns)):
        return [camelcase_to_underscore(x) for x in temp_columns]

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
    return [camelcase_to_underscore(x) for x in result_columns]


def camelcase_to_underscore(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _is_empty_line(line):
    """Tests if a list of strings is empty for example ["", ""] or []
    """
    return [x for x in line if x.strip() != ""] == []


def _listify(colstr):
    """If s is a comma or space separated string turn it into a list"""
    if isinstance(colstr, str):
        if ',' in colstr:
            return [x.strip() for x in colstr.split(',')]
        else:
            return [x for x in colstr.split(' ') if x]
    else:
        return colstr


def _peek_first(seq):
    """Returns a tuple (first_item, it)

    'it' is untouched, first_item is pushed back to be exact
    """
    seq = iter(seq)
    first_item = next(seq)
    return first_item, chain([first_item], seq)


def _create_statement(name, colnames):
    """create table if not exists foo (...)

    Every type is numeric.
    Table name and column names are all lower case.
    """
    schema = ', '.join([col.lower() + ' ' + 'numeric' for col in colnames])
    return "create table if not exists %s (%s)" % (name.lower(), schema)


def _insert_statement(name, ncol):
    "insert into foo values (?, ?, ?, ...)"
    qmarks = ', '.join(['?'] * ncol)
    return "insert into %s values (%s)" % (name, qmarks)


def _select_statement(query, cols=None):
    """If query is just one word, then it is transformed to a select stmt
    or leave it
    """
    cols = cols or '*'
    if len(query.strip().split(' ')) == 1:
        return "select %s from %s" % (', '.join(_listify(cols)), query)
    return query
