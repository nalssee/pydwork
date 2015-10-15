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
__all__ = ['dbopen', 'Row', 'gby', 'gflat', 'load_csv', 'load_xl',
           'chunkn', 'add_header', 'del_header']


import sqlite3
import csv
import tempfile
import openpyxl
import re
import fileinput
import pandas as pd

from collections import Counter
from contextlib import contextmanager
from itertools import groupby, islice, chain, zip_longest


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
    No need to redefine  __getattr__, __setattr__.
    """
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
        query = _select_statement(query)
        if query.strip().partition(' ')[0].upper() != "SELECT":
            raise ValueError("use 'run' for ", query)
        rows = self._cursor.execute(query, args)
        columns = [c[0] for c in rows.description]
        for row1 in rows:
            row = Row()
            for col, val in zip(columns, row1):
                setattr(row, col, val)
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
            rows = self._cursor.execute(_select_statement(query), args)
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


#  Useful for building portfolios
def chunkn(seq, n):
    """Makes n chunks from a seq, each about the same size.
    """
    size = len(list(seq)) / n
    last = 0.0

    i = 0
    while last < len(seq):
        yield i, seq[int(last):int(last + size)]
        last += size
        i += 1


# Some files don't have a header
def add_header(header, filename):
    """Adds a header line to an existing file.
    """
    for line in fileinput.input(filename, inplace=True):
        if fileinput.isfirstline():
            print(header)
        print(line, end='')


def del_header(filename, n=1):
    """Delete n lines from a file
    """
    for line_number, line in enumerate(fileinput.input(filename, inplace=True)):
        if line_number >= n:
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
            n = cnt_copy[col] - cnt[col]
            result_columns.append(col + str(n))
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


def _insert_statement(name, n):
    qs = ', '.join(['?'] * n)
    return "insert into %s values (%s)" % (name, qs)


def _select_statement(query):
    """If query is just one word, then it is transformed to a select stmt
    or leave it
    """
    if len(query.strip().split(' ')) == 1:
        return "select * from " + query
    return query
