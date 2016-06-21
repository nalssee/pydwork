
import os
import sys
import unittest
from itertools import islice

TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)

from pydwork.sqlplus import *

set_workspace(os.path.join(TESTPATH, 'data'))

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
                rows = conn.reel(
                    "select * from first_char order by sp1, sl desc")
                for rs in gby(rows, 'sp1'):
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
            for rs in gby(conn.reel("select * from first_char"), []):
                # the entire data sample is 150
                self.assertEqual(len(rs), 150)

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
            iris = reel('iris.csv', header="no sl sw pl pw sp")

            def first2group():
                for rs in islice(gby(iris, 'sp'), 2):
                    yield from rs
            conn.show(first2group, filename='sample.csv')
            # each group contains 50 rows, hence 100
            self.assertEqual(len(list(reel('sample.csv'))), 100)

    def test_column_case(self):
        with dbopen(':memory:') as conn:
            conn.run("create table Foo (a int, B real)")
            conn.run("insert into foo values (10, 20.2)")
            # table name is case-insensitive
            # but columns names are not
            rows = list(conn.reel('fOo'))

            with self.assertRaises(AttributeError):
                rows[0].b
            self.assertEqual(rows[0].B, 20.2)

            # save it
            conn.save(conn.reel('foo'), name='foo1')
            rows = list(conn.reel('foo1'))

            # now it's lower cased, since it is saved once
            with self.assertRaises(AttributeError):
                rows[0].B
            self.assertEqual(rows[0].b, 20.2)

    def test_order_of_columns(self):
        with dbopen(':memory:') as conn:
            row = next(reel('iris.csv'))
            self.assertEqual(row.columns,
                             ['temp', 'sepal_length', 'sepal_width',
                              'petal_length', 'petal_width', 'species'])
            conn.save(reel('iris.csv'), 'iris')
            row = next(conn.reel('iris'))
            self.assertEqual(row.columns,
                             ['temp', 'sepal_length', 'sepal_width',
                              'petal_length', 'petal_width', 'species'])

    def test_adjoin_disjoin(self):
        with dbopen(':memory:') as conn:
            def unsafe():
                for rs in gby(reel('iris.csv'), 'species'):
                    rs[0].first = 'yes'
                    yield from rs
            with self.assertRaises(Exception):
                conn.save(unsafe)

            # no need to use del anymore here
            @disjoin('sepal_length, sepal_width, temp')
            @adjoin('first, second, third')
            def safe():
                for rs in gby(reel('iris.csv'), 'species'):
                    rs[0].first = 'yes'
                    rs[1].second = 'yes'
                    yield from rs

            # No error
            conn.save(safe)

            r1, r2, *rs = safe()
            self.assertEqual(r1.columns, r2.columns)
            for r in rs:
                self.assertEqual(r1.columns, r.columns)

            self.assertEqual([r1.first, r1.second, r1.third], ['yes', '', ''])
            self.assertEqual([r2.first, r2.second, r2.third], ['', 'yes', ''])

    def test_todf(self):
        with dbopen(':memory:') as conn:
            conn.save(reel('iris.csv'), name='iris')
            for rs in gby(conn.reel('iris'), 'species'):
                self.assertEqual(todf(rs).shape, (50, 6))

    def test_torows(self):
        "Yield pandas data frames and they are flattened again"
        with dbopen(':memory:') as conn:
            conn.save(reel('iris.csv'), name='iris')

            # do not use adjoin or disjoin. it's crazy
            def length_plus_width():
                for rs in gby(conn.reel('iris'), 'species'):
                    df = todf(rs)
                    df['sepal'] = df.sepal_length + df.sepal_width
                    df['petal'] = df.petal_length + df.petal_width
                    del df['sepal_length']
                    del df['sepal_width']
                    del df['petal_length']
                    del df['petal_width']
                    yield from torows(df)

            conn.save(length_plus_width)
            iris = list(conn.reel('iris'))
            for r1, r2 in zip(iris, conn.reel('length_plus_width')):
                a = round(r1.sepal_length + r1.sepal_width, 2)
                b = round(r2.sepal, 2)
                self.assertEqual(a, b)
                c = round(r1.petal_length + r1.petal_width, 2)
                d = round(r2.petal, 2)
                self.assertEqual(c, d)


unittest.main()
