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

    def test_gby(self):
       with dbopen(':memory:') as c:
            # if the string ends with '.csv'
            c.show('iris.csv', 5)

            def first_char(r):
                r.sp1 = r.species[:1]
                return r

            c.save('iris', fn=first_char, name='first_char')

            def top20_sl():
                for rs in c.reel(
                    "select * from first_char order by sp1, sepal_length desc",
                    group='sp1'):
                    yield from rs[:20]

            c.save(top20_sl)

            print("\nYou should see the same two tables")
            print("==========")
            c.show("select col, sepal_length from top20_sl", n=3)
            print("----------")
            c.show(top20_sl, n=3, cols='col, sepal_length')
            print("==========")

            r0, r1 = list(
                c.reel("""select avg(sepal_length) as slavg
                from top20_sl group by sp1
                """))
            self.assertEqual(round(r0.slavg, 3), 5.335)
            self.assertEqual(round(r1.slavg, 3), 7.235)

            # gby with empty list group
            # All of the rows in a table is grouped.
            self.assertEqual(len(list(c.reel('first_char'))), 150)
            # list_tables, in alphabetical order
            self.assertEqual(c.tables, ['first_char', 'top20_sl'])

            # get the whole rows
            for rs in c.reel('top20_sl', group=lambda r: 1):
                self.assertEqual(len(rs), 40)

    def test_run_over_run(self):
        with dbopen(':memory:') as conn:
            conn.save("iris", name="iris1")
            conn.save("iris", name="iris2")
            a = conn.reel("select * from iris1 where species='setosa'")
            b = conn.reel("select * from iris2 where species='versicolor'")
            self.assertEqual(next(a).species, 'setosa')
            self.assertEqual(next(b).species, 'versicolor')
            # now you iterate over 'a' again and you may expect 'setosa'
            # to show up
            # but you'll see 'versicolor'
            # it doesn't matter you iterate over a or b
            # you simply iterate over the most recent query.
            self.assertEqual(next(a).species, 'versicolor')
            self.assertEqual(next(b).species, 'versicolor')

    def test_del(self):
        """tests column deletion
        """
        with dbopen(':memory:') as conn:
            conn.save('co2')

            def co2_less(*col):
                """remove columns"""
                for r in conn.reel('co2'):
                    for c in col:
                        del r[c]
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
        with dbopen(':memory:') as c:
            c.save('iris')
            iris = c.reel('iris', group='species')

            def first2group():
                for rs in islice(iris, 2):
                    yield from rs

            c.show(first2group, filename='sample.csv', n=None)

            # each group contains 50 rows, hence 100
            c.save('sample')
            self.assertEqual(len(list(c.reel('sample'))), 100)
            os.remove(os.path.join('data', 'sample.csv'))

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
        with dbopen(':memory:') as c:
            c.save('iris')
            row = next(c.reel('iris'))
            self.assertEqual(row.columns,
                             ['col', 'sepal_length', 'sepal_width',
                              'petal_length', 'petal_width', 'species'])

    def test_unsafe_save(self):
        with dbopen(':memory:') as c:
            c.save('iris')
            def unsafe():
                for rs in c.reel('iris', group='species'):
                    rs[0].a = 'a'
                    yield rs[0]
                    for r in rs[1:]:
                        r.b = 'b'
                        yield r
            # when rows are not alike, you can't save it
            with self.assertRaises(Exception):
                conn.save(unsafe)

    def test_todf(self):
        with dbopen(':memory:') as conn:
            conn.save('iris')
            for rs in conn.reel('iris', group='species'):
                self.assertEqual(rs.df().shape, (50, 6))


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

        with self.assertRaises(Exception):
            r1.a

        with self.assertRaises(Exception):
            del r1.a

        del r1.y

        self.assertEqual(r1.columns, ['x', 'z'])
        self.assertEqual(r1.values, [10, 39.2])

        r1.x *= 10
        r1.z = r1.x - r1.z
        self.assertEqual(r1.values, [r1.x, r1.z])

    def test_row2(self):
        r1 = Row()
        self.assertEqual(r1.columns, [])
        self.assertEqual(r1.values, [])

        r1['x'] = 10
        r1['y'] = 'abc'
        r1['z'] = 39.2

        self.assertEqual(r1.columns, ['x', 'y', 'z'])
        self.assertEqual(r1.values, [10, 'abc', 39.2])

        with self.assertRaises(Exception):
            r1['a']

        with self.assertRaises(Exception):
            del r1['a']

        del r1['y']

        self.assertEqual(r1.columns, ['x', 'z'])
        self.assertEqual(r1.values, [10, 39.2])

        r1['x'] *= 10
        r1['z'] = r1['x'] - r1['z']
        self.assertEqual(r1.values, [r1['x'], r1['z']])


class TestMisc(unittest.TestCase):
    def test_prepend_header(self):
        # since prepend_header is a util you need to pass the full path
        iris2 = os.path.join('data', 'iris2.csv')
        with dbopen(':memory:') as c:
            c.save('iris')
            c.write(c.reel('iris'), 'iris2.csv')
            prepend_header(iris2, 'cnt, sl, sw, pl, pw, sp', drop=20)
            c.drop('iris2')
            c.save('iris2')
            first = next(c.reel('iris2'))
            self.assertEqual(first.cnt, 20)

            c.write(c.reel('iris'), 'iris2.csv')
            prepend_header(iris2, 'cnt, sl, sw, pl, pw, sp', drop=1)
            c.drop('iris2')
            c.save('iris2')
            first = next(c.reel('iris2'))
            self.assertEqual(first.cnt, 1)

            c.write(c.reel('iris'), 'iris2.csv')
            prepend_header(iris2, 'cnt, sl, sw, pl, pw, sp', drop=0)
            c.drop('iris2')
            c.save('iris2')
            first = next(c.reel('iris2'))
            self.assertEqual(first.cnt, 'col')
            self.assertEqual(first.sl, 'sepal_length')

            c.write(c.reel('iris'), 'iris2.csv')
            # simply drop the first 5 lines, and do nothing else
            prepend_header(iris2, header=None, drop=5)
            # don't drop any and just write the header
            prepend_header(iris2, header='cnt, sl, sw, pl, pw, sp', drop=0)
            c.drop('iris2')
            c.save('iris2')
            first = next(c.reel('iris2'))
            self.assertEqual(first.cnt, 5)

            os.remove(iris2)

    def test_dup_columns(self):
        with dbopen(':memory:') as c:
            c.save('iris')

            with self.assertRaises(Exception):
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
            time.sleep(0.001)
            return x

        start = time.time()
        xs = list(pmap(func, range(10000), chunksize=3, nworkers=8))
        self.assertEqual(xs, list(range(10000)))
        end = time.time()
        self.assertTrue((end - start) < 2)

        # thread version
        start = time.time()
        xs = list(pmap(func, range(10000), chunksize=3, nworkers=8,
                       parallel=False))
        self.assertEqual(xs, list(range(10000)))
        end = time.time()
        self.assertTrue((end - start) < 2)

        def func2(x):
            if x == 4:
                10 / 0
            else:
                return x

        # you must see zero division error message
        # but still you should get 5 elements
        self.assertEqual(list(pmap(func2, range(100), nworkers=2,
                                   parallel=True)),
                         [0, 1, 2, 3])
        self.assertEqual(list(pmap(func2, range(100), nworkers=2,
                                   parallel=False)),
                         [0, 1, 2, 3])

        # first arg for each func can be passed
        def func3(a, x):
            return a + x

        self.assertEqual(list(pmap(func3, '12345',
                                   fargs=['A', 'B'], parallel=True)),
                         ['A1', 'B2', 'A3', 'B4', 'A5'])
        self.assertEqual(list(pmap(func3, '12345',
                                   fargs=['A', 'B'], parallel=False)),
                         ['A1', 'B2', 'A3', 'B4', 'A5'])



class TestRows(unittest.TestCase):
    def test_rows1(self):
        with dbopen(':memory:') as c:
            c.save('iris')
            iris = c.rows('iris')
            self.assertTrue(isinstance(iris[0], Row))
            self.assertTrue(hasattr(iris[2:3], 'order'))
            # hasattr doesn't work correctly for Row
            self.assertFalse('order' in dir(iris[2]))
            del iris[3:]
            self.assertTrue(hasattr(iris, 'order'))
            self.assertEqual(iris['sepal_length, sepal_width, species'][2][2],
                             'setosa')
            with self.assertRaises(Exception):
                iris['one'] = [1]
            iris['one, two'] = [[1, 2] for _ in range(3)]
            self.assertEqual(iris['one, two'], [[1, 2] for _ in range(3)])
            del iris['one, col']
            self.assertTrue(hasattr(iris, 'order'))
            self.assertEqual(len(iris[0].columns), 6)

            # append heterogeneuos row
            iris.append(Row())
            with self.assertRaises(Exception):
                iris.df()
            iris[:3].df()

            with self.assertRaises(Exception):
                c.save(iris, 'iris_sample')
            c.save(iris[:3], 'iris_sample')

    def test_rows2(self):
        with dbopen(':memory:') as c:
            c.save('iris')
            iris = c.rows('iris')
            # order is destructive
            iris.order('sepal_length, sepal_width', reverse=True)
            self.assertEqual(iris[0].col, 132)
            self.assertEqual(iris[1].col, 118)
            self.assertEqual(iris[2].col, 136)

            col1 = iris.equals('species', 'versicolor')[0].col

            self.assertEqual(col1, 51)
            # filter is non-destructive
            self.assertEqual(iris[0].col, 132)

            self.assertEqual(len(next(iris.group('species'))), 12)

            # just because..
            sum = 0
            for rs in iris.group('species'):
                sum += len(rs)
            self.assertEqual(sum, 150)

            # iris = Rows(c.reel('iris'))

            self.assertEqual(len(iris.ge('sepal_length', 7.0)), 13)
            self.assertEqual(len(iris.le('sepal_length', 7.0)), 138)
            self.assertEqual(len(iris.fromto('sepal_length', 5.0, 5.0)), 10)
            self.assertEqual(len(iris.num('species')), 0)
            self.assertEqual(len(iris.text('species')), 150)

            self.assertEqual(len(iris.num('sepal_length, sepal_width')), 150)
            self.assertEqual(len(iris.contains('species',
                                               'versicolor, virginica')),
                             100)
            self.assertEqual(len(iris.contains('sepal_length', 5.0)), 10)

            rs = []
            for x in range(10):
                r = Row()
                r.x = x
                rs.append(r)
            c.save(rs, 'temp')
            rs = c.rows('temp')
            self.assertEqual(rs.truncate('x', 0.2)['x'],
                             [2, 3, 4, 5, 6, 7])

    def test_describe(self):
        with dbopen(':memory:') as c:
            c.save('iris')
            iris = c.rows('iris')
            self.assertTrue('petal_width' in iris[0].columns)
            for g in iris.group('species'):
                df = g.df('sepal_length, sepal_width')
                summary = df.describe()
                self.assertFalse('petal_width' in dir(summary))
                # you can plot it
                # df.plot.scatter(x='sepal_length', y='sepal_width')
                # plt.show()


class TestUserDefinedFunctions(unittest.TestCase):
    def test_simple(self):
        # isnum, istext, yyyymm
        with dbopen(':memory:') as c:
            # fama french 5 industry portfolios
            c.save('indport')
            c.run(
                """
                create table if not exists indport1 as
                select *, substr(date, 1, 4) as yyyy,
                substr(date, 1, 6) as yyyymm,
                case
                when cnsmr >= 0 then 1
                else 'neg'
                end as sign_cnsmr
                from indport
                """)

            na = len(c.rows("select * from indport1 where isnum(sign_cnsmr)"))
            nb = len(c.rows("select * from indport1 where istext(sign_cnsmr)"))
            nc = len(c.rows("select * from indport1"))
            self.assertEqual(na + nb, nc)

            r = next(c.reel(
                """
                select *, yyyymm(substr(date, 1, 6), 12) as yyyymm1,
                yyyymmdd(date, 365) as yyyymmdd1
                from indport
                where date >= 20160801
                """))
            self.assertEqual(r.yyyymm1, 201708)
            self.assertEqual(r.yyyymmdd1, 20170801)


class TestMpairs(unittest.TestCase):
    def test_mpairs(self):
        # lists, iterators are OK
        xs = iter([2, 4, 7, 9, 10, 11, 21])
        ys = [1, 3, 4, 9, 10, 21, 100]
        result = []
        for a, b in mpairs(xs, ys, lambda x: x):
            result.append(a)
        self.assertEqual(result, [4, 9, 10, 21])


class TestOLS(unittest.TestCase):
    def test_ols(self):
        with dbopen(':memory:') as c:
            c.save('iris')
            for rs in c.reel('iris', group='species'):
                result =rs.ols('sepal_length ~ petal_length + petal_width')
                # maybe you should test more here
                self.assertEqual(result.nobs, 50)
                self.assertEqual(len(result.params), 3)


unittest.main()
