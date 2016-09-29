import os
import sys
import unittest
from itertools import islice
import time

TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)

from pydwork.sqlplus import *
from pydwork.util import mpairs, isnum, istext, yyyymm, yyyymmdd, \
    prepend_header, pmap

set_workspace('data')

print('________________________________________________________________')

# This should work as a tutorial as well.
print("\nNo need to read the following")
print("Simply skim through, and recognize if it's not too weird\n\n")


def fillin(line, n):
    """For invalid line handling"""
    if len(line) < n:
        return line + [''] * (n - len(line))
    if len(line) > n:
        return line[:n]
    return line


def count(conn, table):
    if isinstance(table, str):
        return sum(1 for _ in conn.reel(table))
    else:
        # seq
        return sum(1 for _ in table)


class Testdbopen(unittest.TestCase):

    def test_loading(self):
        with dbopen(':memory:') as conn:
            with self.assertRaises(ValueError):
                # column number mismatch, notice pw is missing
                next(reel('iris.csv', header="no sl sw pl species"))
            # when it's loaded, it's just an iterator of objects
            # with string only properties. No type guessing is attempted.
            conn.save(reel('iris.csv', header="no sl sw pl pw species"),
                      name="iris")

    def test_gby(self):
        """Just a dumb presentation to show how 'gby' works.
        """
        with dbopen(':memory:') as conn:
            def first_char():
                "make a new column with the first charactor of species."
                for r in reel('iris.csv', header="no sl sw pl pw species"):
                    # Since r is just an object you can simply add new columns
                    # or delete columns as you'd do with objects.

                    # Each property is either a string, integer or real.
                    r.sp1 = r.species[:1]
                    yield r
           # function name just becomes the table name
            conn.save(first_char)

            def top20_sl():
                for rs in conn.reel(
                    "select * from first_char order by sp1, sl desc",
                    group='sp1'):
                    yield from rs[:20]

            conn.save(top20_sl)

            print("\nYou should see the same two tables")
            print("==========")
            conn.show("select no, sl from top20_sl", n=3)
            print("----------")
            conn.show(top20_sl, n=3, cols='no, sl')
            print("==========")

            r0, r1 = list(
                conn.reel("""select avg(sl) as slavg
                from top20_sl group by sp1
                """))
            self.assertEqual(round(r0.slavg, 3), 5.335)
            self.assertEqual(round(r1.slavg, 3), 7.235)

            # gby with empty list group
            # All of the rows in a table is grouped.
            self.assertEqual(len(Rows(conn.reel('first_char'))), 150)
            # list_tables, in alphabetical order
            self.assertEqual(conn.tables, ['first_char', 'top20_sl'])

    def test_run_over_run(self):
        with dbopen(':memory:') as conn:
            conn.save(reel("iris.csv",
                           header="no,sl,sw,pl,pw,sp"), name="iris1")
            conn.save(reel("iris.csv",
                           header="no,sl,sw,pl,pw,sp"), name="iris2")
            a = conn.reel("select * from iris1 where sp='setosa'")
            b = conn.reel("select * from iris2 where sp='versicolor'")
            self.assertEqual(next(a).sp, 'setosa')
            self.assertEqual(next(b).sp, 'versicolor')
            # now you iterate over 'a' again and you may expect 'setosa'
            # to show up
            # but you'll see 'versicolor'
            # it doesn't matter you iterate over a or b
            # you simply iterate over the most recent query.
            self.assertEqual(next(a).sp, 'versicolor')
            self.assertEqual(next(b).sp, 'versicolor')

    def test_del(self):
        """tests column deletion
        """
        with dbopen(':memory:') as conn:
            conn.save(reel('co2.csv'), name='co2')

            def co2_less(*col):
                """remove columns"""
                co2 = conn.reel("select * from co2")
                for r in co2:
                    for c in col:
                        delattr(r, c)
                    yield r
            print('\nco2 table')
            print('=============================================')
            conn.show("co2", n=2)
            print('=============================================')
            print("\nco2 table without plant and conc")
            print('=============================================')
            # of course you can call conn.show(co2_less('plant', 'conc'), n=2)
            conn.show(co2_less, args=('plant', 'conc'), n=2)
            print('=============================================')
            conn.save(co2_less, args=('plant', 'conc'))

    def test_saving_csv(self):
        with dbopen(':memory:') as conn:
            iris = reel('iris.csv', header="no sl sw pl pw sp", group='sp')

            def first2group():
                for rs in islice(iris, 2):
                    yield from rs
            conn.show(first2group, filename='sample.csv', n=None)
            # each group contains 50 rows, hence 100
            self.assertEqual(len(list(reel('sample.csv'))), 100)
            os.remove(os.path.join(get_workspace(), 'sample.csv'))

    def test_column_case(self):
        with dbopen(':memory:') as conn:
            conn.run("create table Foo (a int, B real)")
            conn.run("insert into foo values (10, 20.2)")
            # table name is case-insensitive
            rows = list(conn.reel('foO'))
            # but columns names are at least in my system, OS X El Capitan
            # I don't know well about it
            self.assertEqual(rows[0].B, 20.2)

    def test_order_of_columns(self):
        with dbopen(':memory:') as conn:
            row = next(reel('iris.csv'))
            self.assertEqual(row.columns,
                             ['col', 'sepal_length', 'sepal_width',
                              'petal_length', 'petal_width', 'species'])
            conn.save(reel('iris.csv'), 'iris')
            row = next(conn.reel('iris'))
            self.assertEqual(row.columns,
                             ['col', 'sepal_length', 'sepal_width',
                              'petal_length', 'petal_width', 'species'])

    def test_unsafe_save(self):
        with dbopen(':memory:') as conn:
            def unsafe():
                for rs in reel('iris.csv', group='species'):
                    rs[0].a = 'a'
                    yield rs[0]
                    for r in rs[1:]:
                        r.b = 'b'
                        yield r

            with self.assertRaises(AssertionError):
                conn.save(unsafe, safe=True)

            # if you don't pass safe as True,
            # 'save' just checks the number of columns
            # so the following doesn't raise any exceptions
            conn.save(unsafe)

    def test_todf(self):
        with dbopen(':memory:') as conn:
            conn.save(reel('iris.csv'), name='iris')
            for rs in conn.reel('iris', group='species'):
                self.assertEqual(todf(rs).shape, (50, 6))

    def test_fromdf(self):
        "Yield pandas data frames and they are flattened again"
        with dbopen(':memory:') as conn:
            conn.save(reel('iris.csv'), name='iris')

            # do not use adjoin or disjoin. it's crazy
            def length_plus_width():
                for rs in conn.reel('iris', group='species'):
                    df = todf(rs)
                    df['sepal'] = df.sepal_length + df.sepal_width
                    df['petal'] = df.petal_length + df.petal_width
                    del df['sepal_length']
                    del df['sepal_width']
                    del df['petal_length']
                    del df['petal_width']
                    yield from fromdf(df)

            conn.save(length_plus_width)
            iris = list(conn.reel('iris'))
            for r1, r2 in zip(iris, conn.reel('length_plus_width')):
                a = round(r1.sepal_length + r1.sepal_width, 2)
                b = round(r2.sepal, 2)
                self.assertEqual(a, b)
                c = round(r1.petal_length + r1.petal_width, 2)
                d = round(r2.petal, 2)
                self.assertEqual(c, d)


class TestRow(unittest.TestCase):
    def test_row(self):
        r1 = Row()
        self.assertEqual(r1.columns, [])
        self.assertEqual(r1.values, [])

        r1.x = 10
        r1.y = 'abc'
        r1.z = 39.2

        self.assertEqual(r1.columns, ['x', 'y', 'z'])
        self.assertEqual(r1.values, [10, 'abc', 39.2])

        with self.assertRaises(AttributeError):
            r1.a

        with self.assertRaises(ValueError):
            del r1.a

        del r1.y

        self.assertEqual(r1.columns, ['x', 'z'])
        self.assertEqual(r1.values, [10, 39.2])

        r1.x *= 10
        r1.z = r1.x - r1.z
        self.assertEqual(r1.values, [r1.x, r1.z])


class TestMisc(unittest.TestCase):
    def test_prepend_header(self):
        # since prepend_header is a util you need to pass the full path
        iris2 = os.path.join(get_workspace(), 'iris2.csv')
        with dbopen(':memory:') as c:
            c.write(reel('iris'), 'iris2.csv')
            prepend_header(iris2, 'cnt, sl, sw, pl, pw, sp', drop=20)
            first = next(reel('iris2.csv'))
            self.assertEqual(first.cnt, '20')

            c.write(reel('iris'), 'iris2.csv')
            prepend_header(iris2, 'cnt, sl, sw, pl, pw, sp')
            first = next(reel('iris2.csv'))
            self.assertEqual(first.cnt, '1')

            c.write(reel('iris'), 'iris2.csv')
            prepend_header(iris2, 'cnt, sl, sw, pl, pw, sp', drop=0)
            first = next(reel('iris2.csv'))
            self.assertEqual(first.cnt, 'col')
            self.assertEqual(first.sl, 'sepal_length')

            c.write(reel('iris'), 'iris2.csv')
            # simply drop the first 5 lines, and do nothing else
            prepend_header(iris2, header=None, drop=5)
            # don't drop any and just write the header
            prepend_header(iris2, header='cnt, sl, sw, pl, pw, sp', drop=0)
            first = next(reel('iris2.csv'))
            self.assertEqual(first.cnt, '5')

            os.remove(iris2)

    def test_duplicates(self):
        with dbopen(':memory:') as c:
            c.save(reel('iris'), name='iris')

            with self.assertRaises(ValueError):
                # there can't be duplicates
                next(c.reel("""
                select species, sepal_length, sepal_length
                from iris
                """))

            r1 = next(c.reel("""
                      select species, sepal_length,
                      sepal_length as sepal_length10
                      from iris
                      """))
            r1.sepal_length10 = r1.sepal_length * 10
            self.assertEqual(r1.values, ['setosa', 5.1, 51.0])

    def test_utilfns(self):
        self.assertTrue(isnum(3))
        self.assertTrue(isnum(-3.32))
        self.assertFalse(isnum('32.3'))

        self.assertFalse(istext(3))
        self.assertTrue(istext('32.3'))

        self.assertEqual(yyyymm(199912, 2), 200002)
        self.assertEqual(yyyymm(199912, -2), 199910)

        self.assertEqual(yyyymmdd(19991231, 2), 20000102)
        self.assertEqual(yyyymmdd(19991231, -2), 19991229)

    def test_pmap(self):
        def func(x):
            time.sleep(0.1)
            return x

        start = time.time()
        xs = list(pmap(func, range(1000), chunksize=1, processes=100))
        print(xs)
        self.assertEqual(xs, list(range(1000)))
        end = time.time()
        # self.assertTrue((end - start) < 2.3)

class TestRows(unittest.TestCase):
    def test_rows(self):
        # You can safely 'Rows' it multiple times of course
        iris = Rows(Rows(Rows(reel('iris'))))
        # order is destructive
        iris.order('sepal_length, sepal_width', reverse=True)
        self.assertEqual(iris[0].col, '132')
        self.assertEqual(iris[1].col, '118')
        self.assertEqual(iris[2].col, '136')

        col1 = iris.filter(lambda r: r.species == 'versicolor')[0].col
        self.assertEqual(col1, '51')
        # filter is non-destructive
        self.assertEqual(iris[0].col, '132')

        self.assertEqual(len(iris.group('species')[0]), 12)

        # just because..
        sum = 0
        for rs in iris.group('species'):
            sum += len(rs)
        self.assertEqual(sum, 150)

        with dbopen(':memory:') as c:
            c.save(reel('iris'), name='iris')
            iris = Rows(c.reel('iris'))

            self.assertEqual(len(iris.ge('sepal_length', 7.0)), 13)
            self.assertEqual(len(iris.le('sepal_length', 7.0)), 138)
            self.assertEqual(len(iris.fromto('sepal_length', 5.0, 5.0)), 10)
            self.assertEqual(len(iris.num('species')), 0)
            self.assertEqual(len(iris.num('sepal_length, sepal_width')), 150)
            self.assertEqual(len(iris.contains('species',
                                               'versicolor, virginica')),
                             100)



class TestUserDefinedFunctions(unittest.TestCase):
    def test_simple(self):
        # isnum, istext, yyyymm
        with dbopen(':memory:') as c:
            # I'm using seeking alpha data here
            c.save(reel('sa'), name='sa')
            self.assertEqual(len(Rows(c.reel("""select * from sa where
                                             tsymbol='MSFT'"""))), 132)
            self.assertEqual(len(Rows(c.reel("""select * from sa where
                                             tsymbol='MSFT' and isnum(n) = 1
                                             """))), 124)
            self.assertEqual(len(Rows(c.reel("""select * from sa where
                                             tsymbol='MSFT' and istext(n) = 1
                                             """))), 8)
            c.drop('sa1')
            c.run("""
                  create table if not exists sa1 as
                  select *, yyyymm(yyyymm, 3) as yyyymm_n3
                  from sa
                  where tsymbol='MSFT'
                  """)

            r0 = next(c.reel('sa1'))
            self.assertEqual(r0.yyyymm_n3, 200504)


class TestMpairs(unittest.TestCase):
    def test_mpairs(self):
        xs = (x for x in [2, 4, 7, 9, 10, 11, 21])
        ys = (y for y in [1, 3, 4, 9, 10, 21, 100])
        result = []
        for a, b in mpairs(xs, ys):
            result.append(a)
        self.assertEqual(result, [4,9,10, 21])


unittest.main()
