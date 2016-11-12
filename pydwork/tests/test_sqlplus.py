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
from pydwork.fi import PRows


set_workspace('data')


class Testdbopen(unittest.TestCase):

    def test_gby(self):
       with dbopen(':memory:') as c:
            # if the string ends with '.csv'
            c.show('iris.csv', 5)

            def first_char(r):
                r.sp1 = r.species[:1]
                return r

            c.save('iris.csv', fn=first_char, name='first_char')

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
            conn.save("iris.csv", name="iris1")
            conn.save("iris.csv", name="iris2")
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
            conn.save('co2.csv')

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
            c.save('iris.csv')
            iris = c.reel('iris', group='species')

            def first2group():
                for rs in islice(iris, 2):
                    yield from rs

            c.write(first2group, 'sample.csv')
            self.assertTrue(os.path.isfile(os.path.join('data', 'sample.csv')))

            # each group contains 50 rows, hence 100
            c.save('sample.csv')
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
            c.save('iris.csv')
            row = next(c.reel('iris'))
            self.assertEqual(row.columns,
                             ['col', 'sepal_length', 'sepal_width',
                              'petal_length', 'petal_width', 'species'])

    def test_unsafe_save(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')
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
            conn.save('iris.csv')
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
            c.save('iris.csv')
            c.write(c.reel('iris'), 'iris2.csv')
            prepend_header(iris2, 'cnt, sl, sw, pl, pw, sp', drop=20)
            c.drop('iris2')
            c.save('iris2.csv')
            first = next(c.reel('iris2'))
            self.assertEqual(first.cnt, 20)

            c.write(c.reel('iris'), 'iris2.csv')
            prepend_header(iris2, 'cnt, sl, sw, pl, pw, sp', drop=1)
            c.drop('iris2')
            c.save('iris2.csv')
            first = next(c.reel('iris2'))
            self.assertEqual(first.cnt, 1)

            c.write(c.reel('iris'), 'iris2.csv')
            prepend_header(iris2, 'cnt, sl, sw, pl, pw, sp', drop=0)
            c.drop('iris2')
            c.save('iris2.csv')
            first = next(c.reel('iris2'))
            self.assertEqual(first.cnt, 'col')
            self.assertEqual(first.sl, 'sepal_length')

            c.write(c.reel('iris'), 'iris2.csv')
            # simply drop the first 5 lines, and do nothing else
            prepend_header(iris2, header=None, drop=5)
            # don't drop any and just write the header
            prepend_header(iris2, header='cnt, sl, sw, pl, pw, sp', drop=0)
            c.drop('iris2')
            c.save('iris2.csv')
            first = next(c.reel('iris2'))
            self.assertEqual(first.cnt, 5)

            os.remove(iris2)

    def test_dup_columns(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')

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

        self.assertEqual(yyyymmdd(19991230, '2 months'), 20000229)
        self.assertEqual(yyyymmdd(19991231, '-2 months'), 19991031)

        self.assertEqual(yyyymm(199912, 2), 200002)
        self.assertEqual(yyyymm(199912, -2), 199910)


        # not 19990531
        self.assertEqual(yyyymmdd(19990430, '1 month'), 19990530)

        self.assertEqual(yyyymmdd(19991231, '2 days'), 20000102)
        self.assertEqual(yyyymmdd(19991231, '-2 days'), 19991229)


class TestPmap(unittest.TestCase):

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

    def test_pmap2(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')

            def func(rs):
                return rs.ols('petal_length ~ petal_width')

            params = []
            for res in pmap(func, c.reel('iris', 'species')):
                params.append(round(res.params[1] * 100))
            self.assertEqual(params, [55.0, 187.0, 65.0])


class TestRows(unittest.TestCase):
    def test_rows1(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')

            iris = c.rows('iris')
            # rows must be iterable
            self.assertEqual(sum(1 for _ in iris), 150)

            self.assertTrue(isinstance(iris[0], Row))
            self.assertTrue(hasattr(iris[2:3], 'order'))
            # hasattr doesn't work correctly for Row

            iris = c.rows('iris')
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

            iris.append(Row())
            with self.assertRaises(Exception):
                c.save(iris, 'iris_sample')
            c.save(iris[:3], 'iris_sample')


    def test_rows2(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')
            iris = c.rows('iris')
            # order is destructive
            iris.order('sepal_length, sepal_width', reverse=True)
            self.assertEqual(iris[0].col, 132)
            self.assertEqual(iris[1].col, 118)
            self.assertEqual(iris[2].col, 136)

            col1 = iris.where(lambda r: r['species'] == 'versicolor')[0].col

            self.assertEqual(col1, 51)
            # where is destructive
            self.assertEqual(iris[0].col, 51)

            iris = c.rows('iris')
            iris.order('sepal_length, sepal_width', reverse=True)

            self.assertEqual(len(next(iris.group('species'))), 12)
            # just because..
            sum = 0
            for rs in iris.group('species'):
                sum += len(rs)
            self.assertEqual(sum, 150)

            iris = c.rows('iris')

            self.assertEqual(len(c.rows('iris').num('species')), 0)
            self.assertEqual(len(c.rows('iris').text('species')), 150)

            self.assertEqual(len(c.rows('iris').num('sepal_length, sepal_width')), 150)
            self.assertEqual(len(c.rows('iris').where(lambda r: r.species in
                                                      ['versicolor', 'virginica'])),
                             100)
            self.assertEqual(len(c.rows('iris').where(lambda r: r.sepal_length == 5.0)), 10)

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
            c.save('iris.csv')
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
            c.save('indport.csv')
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
                select *, yyyymmdd(date, '12 months') as yyyymm1,
                yyyymmdd(date, '365 days') as yyyymmdd1
                from indport
                where date >= 20160801
                """))
            self.assertEqual(r.yyyymm1, 20170801)
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

    def test_mpairs_with_double_dbs(self):
        with dbopen(':memory:') as c1, dbopen(':memory:') as c2:
            c1.save('iris.csv')
            c2.save('co2.csv')

            seq1 = c1.reel(
                """
                select *, round(petal_length) as key from iris
                order by key
                """, group='key')
            seq2 = c2.reel(
                """
                select *, round(uptake / 10) as key from co2
                order by key
                """, group='key')

            lengths = []

            for a, b in mpairs(seq1, seq2, lambda rs: rs[0].key):
                lengths.append((len(a), len(b)))

            self.assertEqual(lengths, [(24, 18), (26, 17),
                                       (3, 24), (26, 24), (43, 1)])


class TestOLS(unittest.TestCase):
    def test_ols(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')
            for rs in c.reel('iris', group='species'):
                result =rs.ols('sepal_length ~ petal_length + petal_width')
                # maybe you should test more here
                self.assertEqual(result.nobs, 50)
                self.assertEqual(len(result.params), 3)


class TestDFMisc(unittest.TestCase):
    def test_describe(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')
            print("IRIS")
            c.desc("""select sepal_length, sepal_width, petal_length,
                   petal_width from iris""")


class TestPRows(unittest.TestCase):
    def setUp(self):
        self.rs1 = []
        for year in range(2001, 2011):
            r = Row()
            r.yyyy = year
            self.rs1.append(r)
        self.rs1 = PRows(self.rs1, 'yyyy')

        self.rs2 = []
        start_month = 200101
        for i in range(36):
            r = Row()
            r.yyyymm = yyyymm(start_month, i)
            self.rs2.append(r)
        self.rs2 = PRows(self.rs2, 'yyyymm')

        self.rs3 = []
        start_date = 20010101
        for i in range(30):
            r = Row()
            r.yyyymmdd = yyyymmdd(start_date, i)
            self.rs3.append(r)
        self.rs3 = PRows(self.rs3, 'yyyymmdd')

        with dbopen(':memory:') as c:
            c.save('indport.csv')
            # to pseudo monthly data
            rs = []
            for rs1 in c.reel('indport order by date', group=lambda r: str(r.date)[0:4]):
                for r in rs1:
                    r.yyyy = int(str(r.date)[0:4])
                    r.fcode = 'A' + str(r.date)[4:]
                    del r.date
                    rs.append(r)
            self.indport = PRows(rs, 'yyyy')

    def test_indi_sort(self):
        with self.assertRaises(ValueError):
            # there are not enough element to make portfolios in 2009
            self.indport.pn('cnsmr', 2).pn('manuf', 3).avg('other')

        self.indport.where(lambda r: r.yyyy < 2009)
        avgport = self.indport.pn('cnsmr', 2).pn('manuf', 3).avg('other')

        self.assertEqual(avgport[0].n, 76)
        self.assertEqual(avgport[1].n, 45)
        self.assertEqual(avgport[2].n, 3)
        self.assertEqual(avgport[3].n, 7)
        self.assertEqual(avgport[4].n, 37)
        self.assertEqual(avgport[5].n, 80)

        self.assertEqual(round(avgport[0].other, 2), -0.63)

        self.indport.pn('cnsmr', 10).avg('other', pncols='pn_cnsmr').pshow('pn_cnsmr')
        self.indport.pn('cnsmr', 2).pn('manuf', 3).avg('other').pshow()

    def test_dpn(self):
        avgport = self.indport.pn('cnsmr', 4).dpn('manuf', 3).dpn('hlth', 2).avg('other')
        for r in avgport.where(lambda r: r.yyyy < 2016):
            self.assertTrue(r.n == 10 or r.n == 11)
        avgport.pshow()

    def test_famac(self):
        fit = self.indport.famac('other ~ cnsmr + manuf + hi_tec + hlth')
        self.assertEqual(round(fit[0].intercept, 2), 0.02)
        self.assertEqual(round(fit[0].cnsmr, 2), 0.44)
        self.assertEqual(round(fit[0].manuf, 2), 0.16)
        self.assertEqual(round(fit[0].hi_tec, 2), 0.03)
        self.assertEqual(round(fit[0].hlth, 2), 0.10)

    def test_rollover(self):
        lengths = []
        for rs0 in self.rs1.roll(3, 2):
            lengths.append(len(rs0))
        self.assertEqual(lengths, [3, 3, 3, 3, 2])

        lengths = []
        for rs0 in self.rs2.where(lambda r: r.yyyymm > 200103).roll(12, 12):
            lengths.append(len(rs0))
        self.assertEqual(lengths, [12, 12, 9])

        lengths = []
        for rs0 in self.rs2.roll(24, 12):
            lengths.append(len(rs0))
        self.assertEqual(lengths, [24, 21, 9])

        lengths = []
        for rs0 in self.rs3.roll('2 weeks', '1 week'):
            lengths.append(len(rs0))
        self.assertEqual(lengths, [14, 14, 14, 9, 2])


unittest.main()
