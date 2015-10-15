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

What you need to know is in unit test code at test/sqlplus_test.py
"""
__all__ = ['dbopen', 'Row', 'gby', 'gflat', 'load_csv', 'load_xl']


import sqlite3
import csv
import tempfile
import openpyxl
import re
import pandas as pd

from collections import Counter
from contextlib import contextmanager
from itertools import groupby, islice, chain, zip_longest, cycle

from pydwork.sqlite_keywords import SQLITE_KEYWORDS


# Most of the time you work with SQL rows.
# You append, delete, replace columns.
# Or you combine rows with similar properties and think of them as a bunch.
# And then later flatten'em if you want.
class Row:
    "SQL row"
    def __init__(self, **kwargs):
        """ex) Row(a=10, b=20) <= two fields row with 'a' and 'b'

        r = Row(a=10, b=20)
        # if you want to add column 'c' and delete 'a'
        r.c = 30
        del r.a
        """
        for k, val in kwargs.items():
            setattr(self, k, val)

    def column_names(self):
        """Returns a list of strings(column names)
        """
        return list(self.__dict__.keys())

    def get_values(self, columns=None):
        """columns must be given so that it doesn't cause ordering problems

        'columns' is a list of strings.
        """
        columns = _listify(columns) or self.column_names()
        return [getattr(self, c) for c in columns]

    def __str__(self):
        return str(self.__dict__)


class Tinfo:
    """Table info
    """
    def __init__(self, table_name, columns):
        """columns : [['col1', 'real'], ['col2', 'int'], ...]
        or 'col1 real, col2 int, ...'
        """
        self.table_name = table_name
        if isinstance(columns, str):
            columns = [c.split() for c in columns.strip().split(',')]
        self.columns = columns

    def cstmt(self):
        """make a create table statement
        """
        return 'create table if not exists {0}(\n{1}\n)' \
            .format(self.table_name,
                    ',\n'
                    .join(['  {0} {1}'
                           .format(cn, ct) for cn, ct in self.columns])
                    .lower())

    def istmt(self):
        """Insert statement
        """
        tname = self.table_name
        return "insert into {0} values ({1})".\
            format(tname, ', '.join(['?'] * len(self.columns)))

    def cols(self):
        "Only column names not types"
        return ', '.join([c for c, _ in self.columns])

    def __str__(self):
        ncols = max([len(c) for c, _ in self.columns])
        return \
            ('\n' + '=' * (ncols + 7)) + '\n' \
            + self.table_name + '\n' \
            + ('-' * (ncols + 7)) + '\n' \
            + '\n'.join([c.ljust(ncols) + ' : ' + t for c, t in self.columns]) \
            + '\n' \
            + ('=' * (ncols + 7))

    def __repr__(self):
        return "Tinfo('%s', %s)" % (self.table_name, self.columns)


class SQLPlus:
    """SQLPlus object works like a sql cursor.
    """
    def __init__(self, dbfile):
        self._dbfile = dbfile
        self._conn = sqlite3.connect(self._dbfile)
        self._cursor = self._conn.cursor()
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
        if query.strip().partition(' ')[0].upper() != "SELECT":
            raise ValueError("use 'run' for ", query)
        rows = self._cursor.execute(query, args)
        columns = [c[0] for c in rows.description]
        for row1 in rows:
            row = Row()
            for col, val in zip(columns, row1):
                setattr(row, col, _det_type(val))
            yield row

    def save(self, seq, name=None, args=()):
        """create a table from an iterator.

        seq is an iterator or a generator function.
        if seq is a generator function and 'name' is not given,
        the function name is going to be the table name.

        'args' are going to be passed as arguments for the generator function
        """
        # if 'it' is a generator function, it is executed to make an iterator
        if hasattr(seq, '__call__'):
            name = name or seq.__name__
            seq = seq(*args)

        if name in self.tables:
            return

        try:
            row0, seq = _peek_first(seq)
        except StopIteration:
            print("Empty sequence")
            return

        colnames = row0.column_names()
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
            for row in gflat(seq):
                vals = [str(v) for v in row.get_values(colnames)]
                fport.write((','.join(vals) + '\n').encode())

            # Every column type is text
            tinfo0 = Tinfo(name, list(zip(colnames, cycle(['text']))))
            self._cursor.execute(tinfo0.cstmt())
            istmt0 = tinfo0.istmt()

            # Now insertion to a DB
            fport.seek(0)
            # read out the first line, header column
            fport.readline()
            for line in fport:
                # line[:-1] because last index indicates '\n'
                try:
                    line_vals = line[:-1].decode().split(',')
                    self._cursor.execute(istmt0, line_vals)
                except Exception as e:
                    print("Failed to save")
                    for col, val in zip_longest(colnames, line_vals):
                        print(col, val)
                    raise ValueError("Invalid line to save")
        self.tables.append(name)


    # Be careful so that you don't overwrite the file
    def show(self, query, args=(), filename=None, n=30):
        """Printing to a screen or saving to a file

        'query' can be either a SQL query string or an iterable.
        'n' is a maximun number of rows to show up,
        if 'query' is the grouped iterator then n is the number of groups
        """
        if isinstance(query, str):
            # just a table name
            if len(query.strip().split(' ')) == 1:
                query = "select * from " + query
            rows = self._cursor.execute(query, args)
            colnames = [c[0] for c in rows.description]

        # then it is an iterable,
        # i.e., a list or an iterator
        else:
            if hasattr(query, '__call__'):
                query = query(*args)
            try:
                row0, rows = _peek_first(query)
            except StopIteration:
                print("Empty sequence")
                return
            colnames = row0.column_names()
            # implicit gflat
            rows = (r.get_values(colnames) for r in gflat(rows))

        if filename:
            # ignore n
            with open(filename, 'w') as fout:
                fout.write(','.join(colnames) + '\n')
                for row1 in rows:
                    row1_str = [str(val) for val in row1]
                    fout.write(','.join(row1_str) + '\n')
        else:
            # show practically all columns
            with pd.option_context("display.max_rows", n), \
                 pd.option_context("display.max_columns", 1000):
                # make use of pandas DataFrame displaying
                print(pd.DataFrame(list(islice(rows, n)), columns=colnames))

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
        splus._conn.commit()
        splus._conn.close()


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
        n = len(columns)
        for line in csv.reader(fin):
            if len(line) != n:
                if _is_empty_line(line):
                    continue
                line = line_fix(line)
                # if it's still not valid
                if len(line) != n:
                    raise ValueError("column number mismatch", columns, line)
            row1 = Row()
            for col, val in zip(columns, line):
                setattr(row1, col, val)
            yield row1


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


# 'grouped row' refers to a Row object
# with all-list properties
def gby(seq, group):
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
    first_group = list(next(g_seq)[1])
    colnames = first_group[0].column_names()

    yield grouped_row(first_group, colnames)
    for _, rows in g_seq:
        yield grouped_row(rows, colnames)


def gflat(seq):
    """Turn an iterator of grouped rows into an iterator of simple rows.
    """
    row0, seq = _peek_first(seq)
    colnames = row0.column_names()
    # Every attribute must be a list
    # No idea how far should I go
    if isinstance(getattr(row0, colnames[0]), list):
        for g_row in seq:
            for vals in zip(*g_row.get_values(colnames)):
                result_row = Row()
                for col, val in zip(colnames, vals):
                    setattr(result_row, col, val)
                yield result_row
    else:
        for row1 in seq:
            yield row1


def get_option(opt_name):
    """get options for displaying
    """
    return OPTIONS[opt_name]


def _det_type(val):
    """Enforce to turn val into number if possible
    """
    # int(3.2) => 3
    # int('3.2') => raises an error
    val = str(val)
    try:
        return int(val)
    except:
        try:
            return float(val)
        except:
            if isinstance(val, str):
                return val
            raise ValueError(val)


def _listify(colstr):
    """If s is a comma or space separated string turn it into a list"""
    if isinstance(colstr, str):
        return [x for x in re.split(',| ', colstr) if x]
    else:
        return colstr


def _gen_valid_column_names(columns):
    """generate valid column names automatically

    ['a', '_b', 'a', 'a1"*c', 'a1c'] => ['a0', 'a_b', 'a1', 'a1c0', 'a1c1']
    """
    temp_columns = []
    for col in columns:
        # save only alphanumeric and underscore
        # and remove all the others
        newcol = re.sub('[^\w]+', '', col)
        if newcol == '':
            newcol = 'temp'
        elif not newcol[0].isalpha():
            newcol = 'a_' + newcol
        elif newcol.upper() in SQLITE_KEYWORDS:
            newcol = 'a_' + newcol
        temp_columns.append(newcol)

    if len(temp_columns) == len(set(temp_columns)):
        return temp_columns

    cnt = {col:n for col, n in Counter(temp_columns).items() if n > 1}
    cnt_copy = dict(cnt)

    result_columns = []
    for col in temp_columns:
        if col in cnt:
            n = cnt_copy[col] - cnt[col]
            result_columns.append(col + str(n))
            cnt[col] -= 1
        else:
            result_columns.append(col)
    return result_columns


def _peek_first(seq):
    """Returns a tuple (first_item, it)

    'it' is untouched, first_item is pushed back to be exact
    """
    seq = iter(seq)
    first_item = next(seq)
    return first_item, chain([first_item], seq)


def _is_empty_line(line):
    """Tests if a list of strings is empty for example ["", ""] or []
    """
    return [x for x in line if x.strip() != ""] == []
