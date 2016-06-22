
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
            conn.show(first2group, filename='sample.csv', n=None)
            # each group contains 50 rows, hence 100
            self.assertEqual(len(list(reel('sample.csv'))), 100)
            os.remove(os.path.join(get_workspace(), 'sample.csv'))

    def test_column_case(self):
        # every query execution is lower cased
        with dbopen(':memory:') as conn:
            conn.run("create table Foo (a int, B real)")
            conn.run("insert into foo values (10, 20.2)")
            # table name is case-insensitive

            rows = list(conn.reel('fOo'))
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


class CustomersAndOrders(unittest.TestCase):
    def setUp(self):
        with dbopen('customers_and_orders.db') as c:
            c.save(read_html_table('customers'), 'customers')
            c.save(read_html_table('orders'), 'orders')

    def tearDown(self):
        import shutil
        summary_dir = os.path.join(get_workspace(), 'summary')
        if os.path.isdir(summary_dir):
            shutil.rmtree(summary_dir)
        os.remove(os.path.join(get_workspace(), 'customers_and_orders.db'))

    def test_read_html_table(self):
        with dbopen('customers_and_orders.db') as c:
            customers = list(c.reel('customers'))
            orders = list(c.reel('orders'))
            self.assertEqual(customers[0].columns,
                             ['customer_id', 'customer_name',
                              'contact_name', 'address', 'city',
                              'postal_code', 'country'])

            self.assertEqual(len(customers), 91)
            self.assertEqual(orders[0].columns,
                             ['order_id', 'customer_id', 'employee_id',
                              'order_date', 'shipper_id'])
            self.assertEqual(len(orders), 196)

            c.summarize(n=17)
            self.assertEqual(len(list(reel('summary/customers'))), 17)
            self.assertEqual(len(list(reel('summary/orders'))), 17)

    def test_drop_it(self):
        with dbopen('customers_and_orders.db') as c:
            def int_postal_code():
                for r in c.reel('customers'):
                    if isinstance(r.postal_code, int):
                        yield r

            def nonint_postal_code():
                for r in c.reel('customers'):
                    if not isinstance(r.postal_code, int):
                        yield r

            self.assertEqual(len(list(int_postal_code())), 66)
            self.assertEqual(len(list(nonint_postal_code())), 25)

            c.save(int_postal_code)
            c.save(nonint_postal_code)
            self.assertEqual(len(c.tables), 4)

            c.summarize()
            summary_dir = os.path.join(get_workspace(), 'summary')

            f1 = os.path.join(summary_dir, 'int_postal_code.csv')
            self.assertTrue(os.path.isfile(f1))
            f2 = os.path.join(summary_dir, 'nonint_postal_code.csv')
            self.assertTrue(os.path.isfile(f2))

            c.drop('int_postal_code')
            self.assertEqual(len(c.tables), 3)
            self.assertFalse(os.path.isfile(f1))

            c.drop('nonint_postal_code')
            self.assertEqual(len(c.tables), 2)
            self.assertFalse(os.path.isfile(f2))

            # must not raise exception
            c.drop('int_postal_code')

    def test_left_join(self):
        with dbopen('customers_and_orders.db') as c:
            c.run("""
            create table customers1 as
            select a.*, b.order_id, b.employee_id, b.order_date, b.shipper_id
            from customers a

            left join orders b
            on a.customer_id = b.customer_id

            """)

            self.assertEqual(len(c.tables), 3)

            def nones():
                for r in c.reel('customers1'):
                    if r.order_id is None:
                        yield r

            self.assertEqual(c.count(nones), 17)
            self.assertEqual(c.count(nones()), 17)

            c.save(nones)
            # # once you have it, you can call count as follows
            self.assertEqual(c.count('nones'), c.count("""
            select * from
            customers1
            where order_id is null
            """))

    def test_gby(self):
        with dbopen('customers_and_orders.db') as c:
            with self.assertRaises(AttributeError):
                # country does not exists in select columns
                for rs in gby(c.reel("""
                select customer_name, postal_code
                from customers
                order by country
                """), 'country'):
                    pass

            total = 0
            for rs in gby(c.reel("""
            select customer_name, postal_code, country
            from customers
            order by country
            """), 'country'):
                df = todf(rs)
                total += df.shape[0]

            self.assertEqual(total, c.count('customers'))

            def major_markets(n):
                """
                find major n countries with lots of customers
                """
                def country_counts():
                    for rs in gby(c.reel("""
                    select *
                    from customers
                    order by country
                    """), 'country'):
                        r = Row()
                        r.country = rs[0].country
                        r.count = len(rs)
                        yield r
                c.save(country_counts)

                countries = []

                for r in islice(c.reel("""
                select * from country_counts
                order by count desc
                """), n):
                    countries.append(r.country)
                return countries

            self.assertEqual(major_markets(5),
                             ['USA', 'France', 'Germany', 'Brazil', 'UK'])

    def test_todf_torows(self):
        with dbopen('customers_and_orders.db') as c:
            query = """
            select *
            from orders
            order by employee_id, order_date
            """

            def orders1():
                for rs in gby(c.reel(query), 'employee_id'):
                    yield from torows(todf(rs))

            self.assertEqual(c.count(orders1), 196)

            with self.assertRaises(AssertionError):
                for a, b in zip(orders1(), c.reel(query)):
                    self.assertEqual(a.customer_id, b.customer_id)

            for a, b in zip(list(orders1()), c.reel(query)):
                self.assertEqual(a.customer_id, b.customer_id)

            for a, b in zip(orders1(), list(c.reel(query))):
                self.assertEqual(a.customer_id, b.customer_id)


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

        with self.assertRaises(KeyError):
            del r1.a

        del r1.y

        self.assertEqual(r1.columns, ['x', 'z'])
        self.assertEqual(r1.values, [10, 39.2])

unittest.main()
