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
__all__ = ['dbopen', 'Row', 'gby', 'gflat', 'load_csv', 'load_xl',
           'chunk', 'add_header', 'del_header', 'adjoin', 'disjoin', 'pick']


import sqlite3
import csv
import tempfile
import openpyxl
import re
import fileinput
import pandas as pd

from collections import Counter
from contextlib import contextmanager
from itertools import groupby, islice, chain
from functools import wraps


# Some of the sqlite keywords are not allowed for column names
# http://www.sqlite.org/sessions/lang_keywords.html
SQLITE_KEYWORDS = [
    "ABORT", "ACTION", "ADD", "AFTER", "ALL", "ALTER", "ANALYZE", "AND",
    "AS", "ASC", "ATTACH", "AUTOINCREMENT", "BEFORE", "BEGIN", "BETWEEN",
    "BY", "CASCADE", "CASE", "CAST", "CHECK", "COLLATE", "COLUMN", "COMMIT",
    "CONFLICT", "CONSTRAINT", "CREATE", "CROSS", "CURRENT_DATE", "CURRENT_TIME",
    "CURRENT_TIMESTAMP", "DATABASE", "DEFAULT", "DEFERRABLE", "DEFERRED",
    "DELETE", "DESC", "DETACH", "DISTINCT", "DROP", "EACH", "ELSE", "END",
    "ESCAPE", "EXCEPT", "EXCLUSIVE", "EXISTS", "EXPLAIN", "FAIL", "FOR",
    "FOREIGN", "FROM", "FULL", "GLOB", "GROUP", "HAVING", "IF", "IGNORE",
    "IMMEDIATE", "IN", "INDEX", "INDEXED", "INITIALLY", "INNER", "INSERT", "INSTEAD",
    "INTERSECT", "INTO", "IS", "ISNULL", "JOIN", "KEY", "LEFT", "LIKE", "LIMIT",
    "MATCH", "NATURAL",
    # no is ok somehow
    # no idea why
    # "NO",
    "NOT",
    "NOTNULL", "NULL", "OF", "OFFSET", "ON", "OR", "ORDER", "OUTER", "PLAN",
    "PRAGMA", "PRIMARY", "QUERY", "RAISE", "REFERENCES", "REGEXP", "REINDEX",
    "RENAME", "REPLACE", "RESTRICT", "RIGHT", "ROLLBACK", "ROW", "SAVEPOINT",
    "SELECT", "SET", "TABLE", "TEMP", "TEMPORARY", "THEN", "TO", "TRANSACTION",
    "TRIGGER", "UNION", "UNIQUE", "UPDATE", "USING", "VACUUM", "VALUES", "VIEW",
    "VIRTUAL", "WHEN", "WHERE"
]


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
        self._dbfile = dbfile
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

        # implicitly gflat
        seq = gflat(seq)
        if nrow:
            seq = islice(seq, nrow)

        if name in self.tables:
            return
        if name is None:
            raise ValueError('name should be passed')

        try:
            row0, seq = _peek_first(seq)
        except StopIteration:
            raise ValueError('Empty sequence')

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
        with tempfile.TemporaryFile() as fport:
            # Write the iterator in a temporary file
            # encode it as binary.
            fport.write((','.join(colnames) + '\n').encode())
            # implicitly flatten
            for row in seq:
                vals = [str(v) for v in row.get_values(colnames)]
                fport.write((','.join(vals) + '\n').encode())

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
                    raise ValueError("Invalid line to save", line_vals)
        self.tables.append(name)

    # Be careful so that you don't overwrite the file
    def show(self, query, args=(), filename=None, n=30, cols=None):
        """Printing to a screen or saving to a file

        'query' can be either a SQL query string or an iterable.
        'n' is a maximun number of rows to show up,
        """
        # so that you can easily maintain code
        # Searching nrows is easier than searching n in editors
        nrows = n

        if isinstance(query, str):
            seq_rvals = self._cursor.execute(_select_statement(query), args)
            colnames = [c[0] for c in seq_rvals.description]

        # then query is an iterator of rows, or a list of rows
        # of course it can be just a generator function of rows
        else:
            rows = query
            if hasattr(rows, '__call__'):
                rows = rows(*args)

            if cols:
                rows = pick(rows, cols)

            try:
                row0, rows = _peek_first(rows)
            except StopIteration:
                print('Empty Sequence')
                return

            colnames = row0.columns
            # implicit gflat
            seq_rvals = (r.get_values(colnames) for r in gflat(rows))

        if filename:
            # ignore n
            with open(filename, 'w') as fout:
                fout.write(','.join(colnames) + '\n')
                for rvals in seq_rvals:
                    fout.write(','.join([str(val) for val in rvals]) + '\n')
        else:
            # show practically all rows, columns.
            with pd.option_context("display.max_rows", nrows), \
                 pd.option_context("display.max_columns", 1000):
                # make use of pandas DataFrame displaying
                print(pd.DataFrame(list(islice(seq_rvals, nrows)), columns=colnames))

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
def gby(seq, group, bind=True):
    """group the iterator by columns

    Based on 'groupby' from itertools

    seq is an iterator
    'group' can be either a function or a list(tuple) of group names.

    if group == [] then group them all
    """
    def grouped_row(rows, columns):
        """Returns a grouped row, from a list of simple rows
        """
        g_row = Row()
        for col in columns:
            setattr(g_row, col, [])
        for row1 in rows:
            for col in columns:
                getattr(g_row, col).append(getattr(row1, col))
        return g_row

    g_seq = groupby(seq, group if hasattr(group, "__call__")
                    else (lambda x: [getattr(x, g) for g in _listify(group)]))
    if bind:
        first_group = list(next(g_seq)[1])
        colnames = first_group[0].columns

        yield grouped_row(first_group, colnames)
        for _, rows in g_seq:
            yield grouped_row(rows, colnames)
    else:
        for _, rows in g_seq:
            yield list(rows)


def gflat(seq):
    """Turn an iterator of grouped rows into an iterator of simple rows.
    """
    def tolist(val):
        if isinstance(val, list):
            return val
        else:
            return [val]

    row0, seq = _peek_first(seq)
    colnames = row0.columns

    for row in seq:
        for vals in zip(*(tolist(getattr(row, col)) for col in colnames)):
            new_row = Row()
            for col, val in zip(colnames, vals):
                setattr(new_row, col, val)
            yield new_row


def pick(seq, cols):
    """map only part of columns passed

    Generator
    Args
        cols: ex) 'col1 col2 col3' or 'col1, col2, col3'
    """
    cols = _listify(cols)
    def partial_row(row):
        "a new row for cols"
        new_row = Row()
        for col in cols:
            setattr(new_row, col, getattr(row, col))
        return new_row
    yield from (partial_row(row) for row in seq)


#  Useful for building portfolios
def chunk(seq, num):
    """Makes num chunks from a seq, each about the same size.
    """
    size = len(list(seq)) / num
    last = 0.0

    i = 0
    while last < len(seq):
        yield i, seq[int(last):int(last + size)]
        last += size
        i += 1


# Some files don't have a header
def add_header(filename, header):
    """Adds a header line to an existing file.
    """
    for line in fileinput.input(filename, inplace=True):
        if fileinput.isfirstline():
            print(header)
        print(line, end='')


def del_header(filename, num=1):
    """Delete n lines from a file
    """
    for line_number, line in enumerate(fileinput.input(filename, inplace=True)):
        if line_number >= num:
            print(line, end='')


def load_csv(csv_file, header=None, line_fix=(lambda x: x)):
    """Loads well-formed csv file, 1 header line and the rest is data

    returns an iterator
    All columns are string, no matter what.
    it's intentional. Types are guessed once it is saved in DB
    """
    with open(csv_file) as fin:
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
                setattr(row1, col, val)
            yield row1


# todo
def load_xl(xl_file, header=None):
    """Loads an Excel file. Only the first sheet

    Basically the same as load_csv.
    """
    def remove_comma(cell):
        """Extracts a comma-removed value from a cell.

        32,120 => 32120
        """
        return str(cell.value).strip().replace(",", "")

    wbook = openpyxl.load_workbook(xl_file)
    sheet_names = wbook.get_sheet_names()
    # only the first sheet
    sheet = wbook.get_sheet_by_name(sheet_names[0])
    rows = sheet.rows
    header = header or [remove_comma(c) for c in rows[0]]
    columns = _gen_valid_column_names(_listify(header))
    for row in rows[1:]:
        cells = []
        for cell in row:
            if cell.value is None:
                cells.append("")
            else:
                cells.append(remove_comma(cell))
        result_row = Row()
        for col, val in zip(columns, cells):
            setattr(result_row, col, val)
        yield result_row


def adjoin(colnames):
    """Decorator to ensure that the rows to have the columns for sure
    """
    def dec(gen):
        @wraps(gen)
        def wrapper(*args, **kwargs):
            for row in gen(*args, **kwargs):
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
        @wraps(gen)
        def wrapper(*args, **kwargs):
            for row in gen(*args, **kwargs):
                for col in _listify(colnames):
                    # whatever it is, just delete it
                    try:
                        delattr(row, col)
                    except:
                        pass
                yield row
        return wrapper
    return dec


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
        return temp_columns

    # Tag numbers to column-names starting from 0 if there are duplicates
    cnt = {col:n for col, n in Counter(temp_columns).items() if n > 1}
    cnt_copy = dict(cnt)

    result_columns = []
    for col in temp_columns:
        if col in cnt:
            result_columns.append(col + str(cnt_copy[col] - cnt[col]))
            cnt[col] -= 1
        else:
            result_columns.append(col)
    return result_columns


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


def _select_statement(query):
    """If query is just one word, then it is transformed to a select stmt
    or leave it
    """
    if len(query.strip().split(' ')) == 1:
        return "select * from " + query
    return query
