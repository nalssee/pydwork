import os
import sys
import unittest
from itertools import groupby, islice

import pandas as pd
from pydwork.sqlplus import *

TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)



def dpath(filename):
    return os.path.join(TESTPATH, 'data', filename)


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
                next(load_csv(dpath('iris.csv'), header="no sl sw pl species"))
            # when it's loaded, it's just an iterator of objects
            # with string only properties. No type guessing is attempted.
            conn.save(load_csv(dpath('iris.csv'),
                               header="no sl sw pl pw species"), name="iris")

    def test_gby(self):
        """Just a dumb presentation to show how 'gby' works.
        """
        with dbopen(':memory:') as conn:
            def first_char():
                "make a new column with the first charactor of species."
                for r in load_csv(dpath('iris.csv'), header="no sl sw pl pw species"):
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
            print("==========")
            conn.show("select no, sl from top20_sl", n=3)
            print("----------")
            conn.show(top20_sl, n=3, cols='no, sl')
            print("==========")

            r0, r1 = list(
                conn.reel("select avg(sl) as slavg from top20_sl group by sp1"))
            self.assertEqual(round(r0.slavg, 3), 5.335)
            self.assertEqual(round(r1.slavg, 3), 7.235)

            # gby with empty list group
            # All of the rows in a table is grouped.
            for g in gby(conn.reel("select * from first_char"), []):
                # the entire data sample is 150
                self.assertEqual(len(g.no), 150)

            # list_tables, in alphabetical order
            self.assertEqual(conn.list_tables(), ['first_char', 'top20_sl'])

            def empty_rows(query):
                for g in gby(conn.reel(query), ["sl"]):
                    if len(g.sl) > 10:
                        yield g

    def test_gflat(self):
        """Tests if applying gby and gflat subsequently yields the original
        """
        with dbopen(':memory:') as conn:
            conn.save(load_csv(dpath("iris.csv"),
                               header="no,sl,sw,pl,pw,sp"), name="iris")
            a = list(conn.reel("select * from iris order by sl"))
            b = list(gflat(gby(conn.reel("select * from iris order by sl"), 'sl')))
            for a1, b1 in zip(a, b):
                self.assertEqual(a1.sl, b1.sl)
                self.assertEqual(a1.pl, b1.pl)

    def test_run_over_run(self):
        with dbopen(':memory:') as conn:
            conn.save(load_csv(dpath("iris.csv"),
                               header="no,sl,sw,pl,pw,sp"), name="iris1")
            conn.save(load_csv(dpath("iris.csv"),
                               header="no,sl,sw,pl,pw,sp"), name="iris2")
            a = conn.reel("select * from iris1 where sp='setosa'")
            b = conn.reel("select * from iris2 where sp='versicolor'")
            self.assertEqual(next(a).sp, 'setosa')
            self.assertEqual(next(b).sp, 'versicolor')
            # now you iterate over 'a' again and you may expect 'setosa' to show up
            # but you'll see 'versicolor'
            # it doesn't matter you iterate over a or b
            # you simply iterate over the most recent query.
            self.assertEqual(next(a).sp, 'versicolor')
            self.assertEqual(next(b).sp, 'versicolor')

    def test_del(self):
        """tests column deletion
        """
        with dbopen(':memory:') as conn:
            conn.save(load_csv(dpath('co2.csv')), name='co2')

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
        import os
        with dbopen(':memory:') as conn:
            iris = load_csv(dpath('iris.csv'), header="no sl sw pl pw sp")
            conn.show(islice(gby(iris, "sp"), 2), filename='sample.csv')
            # each group contains 50 rows, hence 100
            self.assertEqual(len(list(load_csv('sample.csv'))), 100)
            os.remove('sample.csv')

    def test_column_case(self):
        with dbopen(':memory:') as conn:
            conn.run("create table Foo (a int, B real)")
            conn.run("insert into foo values (10, 20.2)")
            rows = list(conn.reel('fOO'))

            self.assertEqual(rows[0].b, '')
            self.assertEqual(rows[0].B, 20.2)

            # save it
            conn.save(conn.reel('foo'), name='foo1')
            rows = list(conn.reel('foo1'))
            # now it's lower cased
            self.assertEqual(rows[0].B, '')
            self.assertEqual(rows[0].b, 20.2)

    def test_add_header(self):
        with dbopen(':memory:') as conn:
            with self.assertRaises(ValueError):
                for r in load_csv(dpath('wierd.csv')):
                    pass
            try:
                add_header(dpath('wierd.csv'), 'a,b,c')
                rows = list(load_csv(dpath('wierd.csv'),
                                     line_fix=lambda x: fillin(x, 3)))
            finally:
                del_header(dpath('wierd.csv'))

            self.assertEqual(len(rows), 7)
            conn.save(rows, name='wierd')

            avals = [r.a for r in conn.reel(
                "select * from wierd order by a")][:3]
            self.assertEqual(avals, [10, 20, 30])

    def test_column_generation(self):
        with dbopen(':memory:') as conn:
            try:
                add_header(dpath('wierd.csv'), 'a,,b,c,c,a,')
                row = next(load_csv(dpath('wierd.csv'),
                                    line_fix=lambda x: fillin(x, 7)))
                self.assertEqual(row.columns,
                                 ['a0', 'temp0', 'b', 'c0', 'c1', 'a1', 'temp1'])
            finally:
                del_header(dpath('wierd.csv'))

            try:
                # in and no are keywords
                # no is ok
                add_header(dpath('wierd.csv'), '_1, in, no, *-*a, a')
                row = next(load_csv(dpath('wierd.csv'),
                                    line_fix=lambda x: fillin(x, 5)))
                self.assertEqual(row.columns,
                                 ['a__1', 'a_in', 'no', 'a0', 'a1'])
            finally:
                del_header(dpath('wierd.csv'))

    def test_order_of_columns(self):
        with dbopen(':memory:') as conn:
            row = next(load_csv(dpath('iris.csv')))
            self.assertEqual(row.columns, ['temp', 'SepalLength', 'SepalWidth',
                                           'PetalLength', 'PetalWidth', 'Species'])
            conn.save(load_csv(dpath('iris.csv')), 'iris')
            row = next(conn.reel('iris'))
            self.assertEqual(row.columns, ['temp', 'sepallength', 'sepalwidth',
                                           'petallength', 'petalwidth', 'species'])

    def test_adjoin_disjoin(self):
        with dbopen(':memory:') as conn:
            def unsafe():
                for rs in gby(load_csv(dpath('iris.csv')), 'Species', bind=False):
                    rs[0].first = 'yes'
                    rs[1].second = 'yes'
                    rs[2].third = 'yes'
                    del rs[2].temp
                    yield from rs
            conn.save(unsafe)
            for r in islice(conn.reel('unsafe'), 5):
                self.assertEqual(r.columns, ['temp', 'sepallength', 'sepalwidth',
                                             'petallength', 'petalwidth', 'species',
                                             'first'
                                             ])

            # no need to use del anymore here
            @disjoin('temp')
            @adjoin('first, second, third')
            def safe():
                for rs in gby(load_csv(dpath('iris.csv')), 'Species', bind=False):
                    rs[0].first = 'yes'
                    rs[1].second = 'yes'
                    rs[2].third = 'yes'
                    yield from rs

            # No error
            conn.save(safe)
            with self.assertRaises(ValueError):
                # temp doesn't exist
                conn.save(pick(safe(), "temp"))

            r1, r2, r3, *_ = safe()
            self.assertEqual([r1.first, r1.second, r1.third], ['yes', '', ''])
            self.assertEqual([r2.first, r2.second, r2.third], ['', 'yes', ''])
            self.assertEqual([r3.first, r3.second, r3.third], ['', '', 'yes'])

    def test_partial_loading(self):
        # You can save only some part of a sequence.
        with dbopen(':memory:') as conn:
            conn.save(gby(load_csv(dpath('iris.csv')),
                          'Species'), n=78, name='setosa')
            self.assertEqual(len(list(conn.reel('setosa'))), 78)

    def test_gflat2(self):
        with dbopen(':memory:') as conn:
            def foo():
                for g in gby(load_csv(dpath('iris.csv')), 'Species'):
                    r = Row()
                    # sometimes just a value
                    r.x = 10
                    # sometimes a list, instead of g.Species[0], just
                    r.s = g.Species
                    yield r
            conn.save(foo)
            self.assertEqual([r.s for r in conn.reel('foo')],
                             ['setosa', 'versicolor', 'virginica'])

    def test_df(self):
        with dbopen(':memory:') as conn:
            conn.save(load_csv(dpath('iris.csv')), name='iris')
            for g in gby(conn.reel('iris'), 'species'):
                self.assertEqual(todf(g).shape, (50, 6))

    def test_gflat3(self):
        "Yield pandas data frames and they are flattened again"
        with dbopen(':memory:') as conn:
            conn.save(load_csv(dpath('iris.csv')), name='iris')

            # do not use adjoin or disjoin. it's crazy
            def length_plus_width():
                for g in gby(conn.reel('iris'), 'species'):
                    df = todf(g)
                    df['sepal'] = df.sepallength + df.sepalwidth
                    df['petal'] = df.petallength + df.petalwidth
                    del df['sepallength']
                    del df['sepalwidth']
                    del df['petallength']
                    del df['petalwidth']
                    yield df

            conn.save(length_plus_width)
            iris_add = list(conn.reel('length_plus_width'))
            for r1, r2 in zip(conn.reel('iris'), iris_add):
                a = round(r1.sepallength + r1.sepalwidth, 2)
                b = round(r2.sepal, 2)
                self.assertEqual(a, b)
                c = round(r1.petallength + r1.petalwidth, 2)
                d = round(r2.petal, 2)
                self.assertEqual(c, d)

unittest.main()
