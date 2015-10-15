import unittest
from pydwork.sqlplus import *
from itertools import islice


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
                next(load_csv('data/iris.csv', header="no sl sw pl species"))
            # when it's loaded, it's just an iterator of objects with string only properties.
            # No type guessing is attempted.
            conn.save(load_csv('data/iris.csv', header="no sl sw pl pw species"), name="iris")

            iris1 = load_csv('data/iris.csv', header="no sl sw pl pw species")
            # Load excel file
            iris2 = load_xl('data/iris.xlsx', header="no sl sw pl pw species")
            for a, b in zip(iris1, iris2):
                self.assertEqual(a.sl, b.sl)
                self.assertEqual(a.pl, b.pl)

    def test_gby(self):
        """Just a dumb presentation to show how 'gby' works.
        """
        with dbopen(':memory:') as conn:
            def first_char():
                "make a new column with the first charactor of species."
                for r in load_csv('data/iris.csv', header="no sl sw pl pw species"):
                    # Since r is just an object you can simply add new columns
                    # or delete columns as you'd do with objects.

                    # Each property is either a string, integer or real.
                    r.sp1 = r.species[:1]
                    yield r
            # function name just becomes the table name
            conn.save(first_char)

            def top20_sl():
                rows = conn.reel("select * from first_char order by sp1, sl desc")
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
            conn.show("top20_sl", num=3)
            print("-------------------------------------")
            # or just see the stream
            conn.show(gflat(top20_sl()), num=3)
            print("=====================================")

            r0, r1 = list(conn.reel("select avg(sl) as slavg from top20_sl group by sp1"))
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
            conn.save(load_csv("data/iris.csv", header="no,sl,sw,pl,pw,sp"), name="iris")
            a = list(conn.reel("select * from iris order by sl"))
            b = list(gflat(gby(conn.reel("select * from iris order by sl"), 'sl')))
            for a1, b1 in zip(a, b):
                self.assertEqual(a1.sl, b1.sl)
                self.assertEqual(a1.pl, b1.pl)

    def test_run_over_run(self):
        with dbopen(':memory:') as conn:
            conn.save(load_csv("data/iris.csv", header="no,sl,sw,pl,pw,sp"), name="iris1")
            conn.save(load_csv("data/iris.csv", header="no,sl,sw,pl,pw,sp"), name="iris2")
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
            conn.save(load_csv('data/co2.csv'), name='co2')
            def co2_less(*col):
                """remove columns"""
                co2 = conn.reel("select * from co2")
                for r in co2:
                    for c in col:
                        delattr(r, c)
                    yield r
            print('\nco2 table')
            print('==============================================================')
            conn.show("select * from co2", num=2)
            print("\nco2 table without plant and number column")
            print("order of columns not preserved")
            print('==============================================================')
            # of course you can call conn.show(co2_less('plant'), n=5)
            conn.show(co2_less, args=('plant', 'no'), num=2)
            print('==============================================================')

            conn.save(co2_less, args=('plant', 'no'))

    def test_saving_csv(self):
        import os
        with dbopen(':memory:') as conn:
            iris = load_csv('data/iris.csv', header="no sl sw pl pw sp")
            conn.show(islice(gby(iris, "sp"), 2), filename='sample.csv')
            # each group contains 50 rows, hence 100
            self.assertEqual(len(list(load_csv('sample.csv'))), 100)
            os.remove('sample.csv')

    def test_column_case(self):
        with dbopen(':memory:') as conn:
            conn.run("create table Foo (a int, B real)")
            conn.run("insert into foo values (10, 20.2)")
            rows = list(conn.reel('fOO'))
            with self.assertRaises(AttributeError):
                print(rows[0].b)
            self.assertEqual(rows[0].B, 20.2)
            # save it
            conn.save(conn.reel('foo'), name='foo1')
            rows = list(conn.reel('foo1'))
            # now it's lower cased
            with self.assertRaises(AttributeError):
                print(rows[0].B)


    def test_add_header(self):

        with dbopen(':memory:') as conn:
            with self.assertRaises(ValueError):
                for r in load_csv('data/wierd.csv'): pass
            try:
                add_header('a,b,c', 'data/wierd.csv')
                rows = list(load_csv('data/wierd.csv', line_fix=lambda x: fillin(x, 3)))
            finally:
                del_header('data/wierd.csv')

            self.assertEqual(len(rows), 7)
            conn.save(rows, name='wierd')

            avals =[r.a for r in conn.reel("select * from wierd order by a")][:3]
            self.assertEqual(avals, [10, 20, 30])

    def test_column_generation(self):
        with dbopen(':memory:') as conn:
            try:
                add_header('a,,b,c,c,a,', 'data/wierd.csv')
                row = next(load_csv('data/wierd.csv', line_fix=lambda x: fillin(x, 7)))
                self.assertEqual(sorted(row.column_names()),
                                 sorted(['a0', 'temp0', 'b', 'c0',
                                         'c1', 'a1', 'temp1']))
            finally:
                del_header('data/wierd.csv')

            try:
                # in and no are keywords
                # no is ok
                add_header('_1, in, no, *-*a, a', 'data/wierd.csv')
                row = next(load_csv('data/wierd.csv', line_fix=lambda x: fillin(x, 5)))
                self.assertEqual(sorted(row.column_names()),
                                 sorted(['a__1', 'a_in', 'no', 'a0', 'a1']))
            finally:
                del_header('data/wierd.csv')


unittest.main()
