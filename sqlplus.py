"""
'sqlite3' based SQL utils
Say if you want to group rows and do some complex jobs,
you can't simply do it with SQL.

And if the data size is so humongous that your system can't load
all up in the memory, you can't easily use pandas either.

For those cases, this module provides a simple generator containing
rows of each group with some other utils
"""

__all__ = ['SQLPlus', 'gby', 'gflat', 'load_csv', 'load_xl']


import sqlite3, csv, tempfile, openpyxl, re
import pandas as pd
from itertools import groupby, islice, chain
from collections import namedtuple
from messytables import CSVTableSet, type_guess
from messytables.types import DecimalType, IntegerType


# CAUTION:
# While a cursor is executing,
# other works using the cursor cannot iterfere the process.
# Be careful especially when you are working with select query.
class SQLPlus:
    """SQLPlus object works like a sql cursor.
    """
    def __init__(self, dbfile):
        self._dbfile = dbfile
        self._conn = sqlite3.connect(self._dbfile)
        self._cursor = self._conn.cursor()

    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        self._conn.commit()
        self._conn.close()

    # args can be a list, a tuple or a dictionary
    def run(self, query, args=[]):
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

        If 'name' is not given, a name of the namedtuple is going to be it.
        """
        if hasattr(it, '__call__'):
            name = name or it.__name__
            it = it(*args)

        it = iter(it)
        try:
            first_row = next(it)
        except StopIteration:
            raise ValueError("Empty rows")

        colnames = first_row.column_names()

        if isinstance(getattr(first_row, colnames[0]), list):
            it = gflat(chain([first_row], it))
        else:
            it = chain([first_row], it)

        with tempfile.TemporaryFile() as f:
            # writing
            f.write((','.join(colnames) + '\n').encode())
            for row in it:
                f.write((','.join([str(getattr(x, c)) for c in colnames]) + '\n').encode())

            # type check
            f.seek(0)
            types = _field_types(f)
            tinfo0 = Tinfo(name, list(zip(colnames, types)))
            self._cursor.execute(tinfo0.cstmt())
            istmt0 = tinfo0.istmt()

            # loading
            f.seek(0)
            # read out the first line
            f.readline()
            for line in f:
                self._cursor.execute(istmt0, line[:-1].decode().split(','))

    # Be careful so that you don't overwrite the file
    def show(self, query, args=[], n=30, filename=None):
        """If 'filename' is given, n is ignored.

        'query' can be either a query string or an iterable.
        """
        if isinstance(query, str):
            query = self._cursor.execute(query, args)
            column_names = [c[0] for c in query.description]
        # if query is not a string, then it is an iterable.
        else:
            query = iter(query)
            first_row = next(query)
            column_names = first_row.column_names()
            rest = list(islice(query, n - 1))
            query = [first_row] + rest

        if filename:
            df = pd.DataFrame(list(query), columns=column_names)
            df.to_csv(filename, index=False)
        else:
            # slicing 1 more so that pandas to show "..." in case there are more
            df = pd.DataFrame(list(islice(query, n)), columns=column_names)
            with pd.option_context("display.max_rows", n):
                with pd.option_context("display.max_columns", 1000):
                    print(df)

    def list_tables(self):
        query = self._cursor.execute("""
        select * from sqlite_master
        where type='table'
        """)
        tables = [row[1] for row in query]
        return sorted(tables)

    def table_info(self, tname):
        # You cannot use parameter here, only certain values are allowed.
        query = self._cursor.execute("pragma table_info({})".format(tname))
        return Tinfo(tname, [[row[1], row[2]] for row in query])


def _listify(s):
    """If s is a comma or space separated string turn it into list"""
    if isinstance(s, str):
        columns = [x for x in re.split(',| ', s) if x]
    elif isinstance(s, list):
        columns = s
    else:
        raise ValueError("header inappropriate", s)
    return columns


# loaded data is string, no matter what!!
def load_csv(csv_file, header=None):
    """Loads well-formed csv file, 1 header line and the rest is data

    returns an iterator
    """
    with open(csv_file) as f:
        first_line = f.readline()
        header = header or first_line
        columns = _listify(header)
        for line in csv.reader(f):
            r =  Row()
            for c, v in zip(columns, line):
                setattr(r, c, v)
            yield r


def load_xl(xl_file, header=None):
    """Loads an Excel file. Only the first sheet
    """
    def remove_comma(cell):
        return str(cell.value).strip().replace(",", "")

    wbook = openpyxl.load_workbook(xl_file)
    sheet_names = wbook.get_sheet_names()
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


def gby(it, group):
    """group the iterator by columns

    'it' is an iterator
    'group' can be either a function or a list(tuple) of group names.

    if group == [] then group them all
    """
    group = _listify(group)

    g_it = groupby(it, group if hasattr(group, "__call__") \
                   else (lambda x: [getattr(x, g) for g in group]))
    first_group = list(next(g_it)[1])
    first_row = first_group[0]
    g_tup = namedtuple("GROUP", first_row._fields)

    yield g_tup(*tr(first_group))
    for _, g in g_it:
        yield g_tup(*tr(g))

# flatten groups
def gflat(it):
    """a stream of a tuple of lists are flattened to a stream of tuples
    """
    it = iter(it)
    try:
        first_group = next(it)
    except StopIteration:
        raise ValueError("Empty Rows")

    tup = namedtuple(first_group.__class__.__name__, first_group._fields)
    for g in chain([first_group], it):
        for r in zip(*g):
            yield tup(*r)

# Table info class
class Tinfo:
    # columns : [['col1', 'real'], ...]
    def __init__(self, table_name, columns):
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
        tname = self.table_name
        n = len(self.columns)
        return "insert into {0} values ({1})".format(tname, ', '.join(['?'] * n))

    def cols(self):
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

# You might need to check how costly this is.
def _field_types(f):
    """Guess field types from CSV

    f is a binary port
    """
    def conv(t):
        if isinstance(t, IntegerType): return "int"
        if isinstance(t, DecimalType): return 'real'
        return 'text'
    tset = CSVTableSet(f)
    row_set = tset.tables[0]
    return [conv(t) for t in type_guess(row_set.sample)]


class Row:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def column_names(self):
        return list(self.__dict__.keys())


# todo: add more tests
# This should work as a tutorial as well.
if __name__ == "__main__":
    import unittest

    class TestSQLPlus(unittest.TestCase):
        def test_loading(self):
            with SQLPlus(':memory:') as conn:
                with self.assertRaises(TypeError):
                    # column number mismatch, notice pw is missing
                    for r in load_csv('iris.csv', header="no sl sw pl species"):
                        print(r)
                # when it's loaded, it's just an iterator of a (named)tuple of strings
                conn.save(load_csv('iris.csv', header="no sl sw pl pw species"), name="iris")

                iris = conn.table_info("iris")
                print("\niris table info", end="")
                print(iris)
                # But once you save it, this program automatically decides types.
                self.assertEqual(iris.table_name, 'iris')
                self.assertEqual(iris.columns, [['no', 'int'], \
                                                ['sl', 'real'], ['sw', 'real'], \
                                                ['pl', 'real'], ['pw', 'real'], \
                                                ['species', 'text']])

        def test_gby(self):
            with SQLPlus(':memory:') as conn:

                def first_char():
                    "make a new column with the first charactor of species."
                    for r in load_csv('iris.csv', header="no sl sw pl pw species"):
                        r.sp1 = r.species[:1]
                        yield r
                conn.save(first_char)

                def top20_sl():
                    rows = conn.run("select * from first_char order by sp1, sl desc")
                    for g in gby(rows, "sp1"):
                        g.no = g.no[:20]
                        yield g
                conn.save(top20_sl, flatten=True)

                print("\nYou should see the same two tables")
                print("=====================================")
                # you can send a query
                conn.show("select * from top20_sl", n=3)
                print("-------------------------------------")
                # or just see the stream
                conn.show(gflat(top20_sl()), n=3)
                print("=====================================")
                # don't forget that you can save it in a file as well
                # and 'n' parameter is ignored.


                r0, r1 = list(conn.run("select avg(sl) as slavg from top20_sl group by sp1"))
                self.assertEqual(round(r0.slavg, 3), 5.335)
                self.assertEqual(round(r1.slavg, 3), 7.235)

                # gby with empty list group
                # All of the rows in a table is grouped.
                for g in gby(conn.run("select * from first_char"), []):
                    self.assertEqual(len(g.no), 150)

                # list_tables
                self.assertEqual(sorted(conn.list_tables()), ['first_char', 'top20_sl'])

                def empty_rows(query):
                    for g in gby(conn.run("select * from first_char order by sl, sw"), "sl"):
                        if len(g.sl) > 10: yield g
                with self.assertRaises(ValueError):
                    conn.save(empty_rows)

    unittest.main()

# import time

# it = (x for x in range(100000000))
# start = time.time()
# def again():
#     for i in it:
#         yield i

# # for i in it:
# #     pass

# for i in again():
#     pass

# end = time.time()
# print("%f" % (end - start,))



# def foo():
#     print("")
#     if True:
#         return it
#     next(it)
#     for i in it:
#         yield i
print('done')
