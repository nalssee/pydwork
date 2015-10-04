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

What you need to know is in unit test code at the end of this file
"""

__all__ = ['SQLPlus', 'Row', 'gby', 'gflat', 'load_csv', 'load_xl']


import sqlite3, csv, tempfile, openpyxl, re
import pandas as pd
from itertools import groupby, islice, chain
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

    def get_values(self, columns):
        """columns must be given so that it doesn't cause ordering problems

        'columns' is a list of strings.
        """
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

        # it can be a list
        it = iter(it)
        try:
            first_row = next(it)
        except StopIteration:
            raise ValueError("Empty rows")

        colnames = first_row.column_names()
        # if the first_rows properties are lists, it is to be flattend
        if isinstance(getattr(first_row, colnames[0]), list):
            it = gflat(chain([first_row], it))
        else:
            it = chain([first_row], it)

        # You can't save the iterator directly because
        # once you execute a table creation query,
        # then the query becomes the most recent query,
        # not the query for the iterator.
        # Which means, if you iterate over the iterator,
        # it iterates over the table creation query.

        # You can see the example at the bottom of this file at the unit test section,
        # 'test_run_over_run'

        # So you save the iterator up in a temporary file
        # and then save the file to a database.
        # In the process column types are checked.
        with tempfile.TemporaryFile() as f:
            # Write the iterator in a temporary file
            # encode it as binary.
            f.write((','.join(colnames) + '\n').encode())
            for row in it:
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

        'n' is a maximun number of rows to show up
        If 'filename' is given, n is ignored.
        """
        if isinstance(query, str):
            query = self._cursor.execute(query, args)
            colnames = [c[0] for c in query.description]
            values = list(islice(query, n))

        # if query is not a string, then it is an iterable,
        # i.e., a list or an iterator
        else:
            query = iter(query)
            first_row = next(query)
            colnames = first_row.column_names()
            values = []
            for r in chain([first_row], islice(query, n - 1)):
                values.append(r.get_values(colnames))

        # make use of pandas' data frame displaying.
        df = pd.DataFrame(values, columns=colnames)
        if filename:
            df.to_csv(filename, index=False)
        else:
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
        first_line = f.readline()
        header = header or first_line
        columns = _listify(header)
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
    it = iter(it)
    try:
        g_row1 = next(it)
    except StopIteration:
        raise ValueError("Empty Rows")
    colnames = g_row1.column_names()

    for gr in chain([g_row1], it):
        for xs in zip(*gr.get_values(colnames)):
            r = Row()
            for c, v in zip(colnames, xs):
                setattr(r, c, v)
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


# This should work as a tutorial as well.
if __name__ == "__main__":
    import unittest

    class TestSQLPlus(unittest.TestCase):
        def test_loading(self):
            with SQLPlus(':memory:') as conn:
                with self.assertRaises(AssertionError):
                    # column number mismatch, notice pw is missing
                    for r in load_csv('iris.csv', header="no sl sw pl species"):
                        print(r)
                # when it's loaded, it's just an iterator of objects with string only properties.
                # No type guessing is attempted.
                conn.save(load_csv('iris.csv', header="no sl sw pl pw species"), name="iris")

                iris = conn.table_info("iris")
                print("\niris table info", end="")
                print(iris)

                # Once you save it, types are automatically determined.
                self.assertEqual(iris.table_name, 'iris')
                # Order of columns can't be predicted.
                # it is determined by the built-in Python hash function.
                self.assertEqual(sorted(iris.columns), \
                                 sorted([['no', 'int'], \
                                         ['sl', 'real'], ['sw', 'real'], \
                                         ['pl', 'real'], ['pw', 'real'], \
                                         ['species', 'text']]))

                iris1 = load_csv('iris.csv', header="no sl sw pl pw species")
                # Load excel file
                iris2 = load_xl('iris.xlsx', header="no sl sw pl pw species")
                for a, b in zip(iris1, iris2):
                    self.assertEqual(a.sl, b.sl)
                    self.assertEqual(a.pl, b.pl)

        def test_gby(self):
            """Just a dumb presentation to show how 'gby' works.
            """
            with SQLPlus(':memory:') as conn:
                def first_char():
                    "make a new column with the first charactor of species."
                    for r in load_csv('iris.csv', header="no sl sw pl pw species"):
                        # Since r is just an object you can simply add new columns
                        # or delete columns as you'd do with objects.

                        # Each property is either a string, integer or real.
                        r.sp1 = r.species[:1]
                        yield r
                # function name just becomes the table name
                conn.save(first_char)

                def top20_sl():
                    rows = conn.run("select * from first_char order by sp1, sl desc")
                    for g in gby(rows, "sp1"):
                        # g is again an object
                        # just each property is a list.
                        # And all properties are of the same length at this point

                        # Say you add a column with 20 items
                        # and the other columns have items larger than that.
                        # Then if you save this iterator in DB,
                        # the other columns are cut in at 20 as well.

                        # Think what zip does. see gflat
                        g.no = g.no[:20]
                        yield g
                # If you are saving grouped objects,
                # they are flattened first
                conn.save(top20_sl)

                print("\nYou should see the same two tables")
                print("=====================================")
                # you can send a query
                conn.show("select * from top20_sl", n=3)
                print("-------------------------------------")
                # or just see the stream
                conn.show(gflat(top20_sl()), n=3)
                print("=====================================")

                r0, r1 = list(conn.run("select avg(sl) as slavg from top20_sl group by sp1"))
                self.assertEqual(round(r0.slavg, 3), 5.335)
                self.assertEqual(round(r1.slavg, 3), 7.235)

                # gby with empty list group
                # All of the rows in a table is grouped.
                for g in gby(conn.run("select * from first_char"), []):
                    # the entire data sample is 150
                    self.assertEqual(len(g.no), 150)

                # list_tables, in alphabetical order
                self.assertEqual(conn.list_tables(), ['first_char', 'top20_sl'])

                def empty_rows(query):
                    for g in gby(conn.run(query), ["sl"]):
                        if len(g.sl) > 10:
                            yield g
                # try to save empty rows
                with self.assertRaises(ValueError):
                    conn.save(empty_rows, args=("select * from first_char order by sl, sw",))

        def test_gflat(self):
            """Tests if applying gby and gflat subsequently yields the original
            """
            with SQLPlus(':memory:') as conn:
                conn.save(load_csv("iris.csv", header="no,sl,sw,pl,pw,sp"), name="iris")
                a = list(conn.run("select * from iris order by sl"))
                b = list(gflat(gby(conn.run("select * from iris order by sl"), "sl")))
                for a1, b1 in zip(a, b):
                    self.assertEqual(a1.sl, b1.sl)
                    self.assertEqual(a1.pl, b1.pl)

        def test_run_over_run(self):
            with SQLPlus(':memory:') as conn:
                conn.save(load_csv("iris.csv", header="no,sl,sw,pl,pw,sp"), name="iris1")
                conn.save(load_csv("iris.csv", header="no,sl,sw,pl,pw,sp"), name="iris2")
                a = conn.run("select * from iris1 where sp='setosa'")
                b = conn.run("select * from iris2 where sp='versicolor'")
                self.assertEqual(next(a).sp, 'setosa')
                self.assertEqual(next(b).sp, 'versicolor')
                # now you iterate over 'a' again and you may expect 'setosa' to show up
                # but you'll see 'versicolor'
                # it doesn't matter you iterate over a or b
                # you simply iterate over the most recent query.
                self.assertEqual(next(a).sp, 'versicolor')
                self.assertEqual(next(b).sp, 'versicolor')

    unittest.main()
