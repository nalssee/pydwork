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

__all__ = ['SQLPlus', 'Row', 'gby', 'gflat', 'load_csv', 'load_xl']


import sqlite3, csv, tempfile, openpyxl, re
import pandas as pd
from itertools import groupby, islice, tee, chain
from messytables import CSVTableSet, type_guess
from messytables.types import DecimalType, IntegerType

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
        for k, v in kwargs.items():
            setattr(self, k, v)

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
        """columns : [['col1', 'real'], ['col2', 'int'], ...] or 'col1 real, col2 int, ...'
        """
        self.table_name = table_name
        if isinstance(columns, str):
            columns = [c.split() for c in columns.strip().split(',')]
        self.columns = columns

    def cstmt(self):
        """make a create table statement
        """
        return 'create table if not exists {0}(\n{1}\n)'\
            .format(self.table_name, ',\n'\
                    .join(['  {0} {1}'.format(cn, ct) for cn, ct in self.columns])).lower()

    def istmt(self):
        """Insert statement
        """
        tname = self.table_name
        n = len(self.columns)
        return "insert into {0} values ({1})".format(tname, ', '.join(['?'] * n))

    def cols(self):
        "Only column names not types"
        return ', '.join([c for c, _ in self.columns])

    def __str__(self):
        n = max([len(c) for c, _ in self.columns])
        return \
            ('\n' + '=' * (n + 7)) + '\n' \
            + self.table_name +'\n' \
            + ('-' * (n + 7)) + '\n' \
            + '\n'.join([c.ljust(n) + ' : ' + t for c, t in self.columns]) + '\n' \
            + ('=' * (n + 7))

    def __repr__(self):
        return "Tinfo('%s', %s)" % (self.table_name, self.columns)


class SQLPlus:
    """SQLPlus object works like a sql cursor.
    """
    def __init__(self, dbfile):
        self._dbfile = dbfile
        self._conn = sqlite3.connect(self._dbfile)
        self._cursor = self._conn.cursor()

    def __enter__(self):
        return self

    def __exit__(self, exe_type, value, trace):
        self._conn.commit()
        self._conn.close()

    # args can be a list, a tuple or a dictionary
    def run(self, query, args=()):
        """Simply executes sql statement

        In case it's 'select' statment,
        an iterator is returned.
        """
        q = self._cursor.execute(query, args)
        command = [x for x in query.split(" ") if not x == ""][0]
        if command.upper() == "SELECT":
            columns = [c[0] for c in q.description]
            for xs in q:
                r = Row()
                for col, val in zip(columns, xs):
                    setattr(r, col, val)
                yield r

    def save(self, it, name=None, args=()):
        """create a table from an iterator.

        'it' is an iterator or a generator function.
        if 'it' is a generator function and 'name' is not given,
        the function name is going to be the table name.

        'args' are going to be passed as arguments for the generator function
        """
        # if 'it' is a generator function, it is executed to make an iterator
        if hasattr(it, '__call__'):
            name = name or it.__name__
            it = it(*args)

        r0, it = _peek_first(it)
        colnames = r0.column_names()
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
        with tempfile.TemporaryFile() as f:
            # Write the iterator in a temporary file
            # encode it as binary.
            f.write((','.join(colnames) + '\n').encode())
            # implicitly flatten
            for row in gflat(it):
                vals = [str(v) for v in row.get_values(colnames)]
                f.write((','.join(vals) + '\n').encode())

            # type check, using 'messytables' package
            f.seek(0)
            types = _field_types(f)
            tinfo0 = Tinfo(name, list(zip(colnames, types)))
            self._cursor.execute(tinfo0.cstmt())
            istmt0 = tinfo0.istmt()

            # Now insertion to a DB
            f.seek(0)
            # read out the first line, header column
            f.readline()
            for line in f:
                # line[:-1] because last index indicates '\n'
                self._cursor.execute(istmt0, line[:-1].decode().split(','))

    # Be careful so that you don't overwrite the file
    def show(self, query, args=(), n=30, filename=None):
        """Printing to a screen or saving to a file

        'query' can be either a SQL query string or an iterable.
        'n' is a maximun number of rows to show up,
        if 'query' is the grouped iterator then n is the number of groups
        """
        if isinstance(query, str):
            query = self._cursor.execute(query, args)
            colnames = [c[0] for c in query.description]
            rows = islice(query, n)

        # then it is an iterable,
        # i.e., a list or an iterator
        else:
            if hasattr(query, '__call__'):
                query = query(*args)
            r0, it = _peek_first(query)
            colnames = r0.column_names()
            # implicit gflat
            rows = (r.get_values(colnames) for r in gflat(islice(it, n)))

        if filename:
            with open(filename, 'w') as f:
                f.write(','.join(colnames) + '\n')
                for r in rows:
                    rstr = [str(r1) for r1 in r]
                    f.write(','.join(rstr) + '\n')
        else:
            # make use of pandas' data frame displaying.
            df = pd.DataFrame(list(rows), columns=colnames)
            # show practically all columns
            with pd.option_context("display.max_rows", n), \
                 pd.option_context("display.max_columns", 1000):
                print(df)

    def list_tables(self):
        """List of table names in the database
        """
        query = self._cursor.execute("""
        select * from sqlite_master
        where type='table'
        """)
        tables = [row[1] for row in query]
        return sorted(tables)

    def table_info(self, tname):
        """Returns a Tinfo object

        'tname' is a string
        """
        # You cannot use parameter here, only certain values(as numbers) are allowed.
        # see Python sqlite3 package manual
        query = self._cursor.execute("pragma table_info({})".format(tname))
        return Tinfo(tname, [[row[1], row[2]] for row in query])


def load_csv(csv_file, header=None):
    """Loads well-formed csv file, 1 header line and the rest is data

    returns an iterator
    All columns are string, no matter what.
    it's intentional. Types are guessed once it is saved in DB
    """
    with open(csv_file) as f:
        first_line = f.readline()[:-1]
        header = header or first_line
        columns = _listify(header)
        assert all(_is_valid_column_name(c) for c in columns), 'Invalid column name'
        for line in csv.reader(f):
            assert len(columns) == len(line), "column number mismatch"
            r = Row()
            for c, v in zip(columns, line):
                setattr(r, c, v)
            yield r


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
    columns = _listify(header)
    assert all(_is_valid_column_name(c) for c in columns), 'Invalid column name'
    for row in rows[1:]:
        cells = []
        for cell in row:
            if cell.value == None:
                cells.append("")
            else:
                cells.append(remove_comma(cell))
        r = Row()
        for c, v in zip(columns, cells):
            setattr(r, c, v)
        yield r


# 'grouped row' refers to a Row object
# with all-list properties
def gby(it, group):
    """group the iterator by columns

    Based on 'groupby' from itertools

    'it' is an iterator
    'group' can be either a function or a list(tuple) of group names.

    if group == [] then group them all
    """
    def grouped_row(rows, columns):
        """Returns a grouped row, from a list of simple rows
        """
        g_row = Row()
        for c in columns:
            setattr(g_row, c, [])
        for r in rows:
            for c in columns:
                getattr(g_row, c).append(getattr(r, c))
        return g_row

    g_it = groupby(it, group if hasattr(group, "__call__") \
                   else (lambda x: [getattr(x, g) for g in _listify(group)]))
    first_group = list(next(g_it)[1])
    columns = first_group[0].column_names()

    yield grouped_row(first_group, columns)
    for _, g in g_it:
        yield grouped_row(g, columns)


def gflat(it):
    """Turn an iterator of grouped rows into an iterator of simple rows.
    """
    r0, it = _peek_first(it)
    colnames = r0.column_names()
    # Every attribute must be a list
    # No idea how far should I go
    if isinstance(getattr(r0, colnames[0]), list):
        for gr in it:
            for xs in zip(*gr.get_values(colnames)):
                r = Row()
                for c, v in zip(colnames, xs):
                    setattr(r, c, v)
                yield r
    else:
        for r in it:
            yield r


# Not sure how costly this is.
# This is what 'messytables' pacakge does.
def _field_types(f):
    """Guess field types from a CSV file

    f is a binary file port
    """
    def conv(t):
        """Once messytables guess what type it is,
        then is is turned into an appropriate sql type,
        i.e., 'int', 'real', or 'text'
        """
        if isinstance(t, IntegerType):
            return "int"
        if isinstance(t, DecimalType):
            return 'real'
        return 'text'
    tset = CSVTableSet(f)
    row_set = tset.tables[0]
    return [conv(t) for t in type_guess(row_set.sample)]


def _listify(s):
    """If s is a comma or space separated string turn it into a list"""
    if isinstance(s, str):
        return [x for x in re.split(',| ', s) if x]
    else:
        return s


def _is_valid_column_name(s):
    return re.match(r'^[A-z]\w*$', s)


def _peek_first(it):
    """Returns a tuple (first_item, it)

    'it' is untouched, first_item is pushed back, to be exact
    """
    it = iter(it)
    first_item = next(it)
    return first_item, chain([first_item], it)
