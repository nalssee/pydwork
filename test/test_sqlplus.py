import unittest
from pydwork.sqlplus import *

# This should work as a tutorial as well.
print("\nNo need to read the following")
print("Simply skim through, and recognize if it's not too weird\n\n")

class TestSQLPlus(unittest.TestCase):
    def test_loading(self):
        with SQLPlus(':memory:') as conn:
            with self.assertRaises(AssertionError):
                # column number mismatch, notice pw is missing
                next(load_csv('data/iris.csv', header="no sl sw pl species"))
            # when it's loaded, it's just an iterator of objects with string only properties.
            # No type guessing is attempted.
            conn.save(load_csv('data/iris.csv', header="no sl sw pl pw species"), name="iris")

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

            iris1 = load_csv('data/iris.csv', header="no sl sw pl pw species")
            # Load excel file
            iris2 = load_xl('data/iris.xlsx', header="no sl sw pl pw species")
            for a, b in zip(iris1, iris2):
                self.assertEqual(a.sl, b.sl)
                self.assertEqual(a.pl, b.pl)

    def test_gby(self):
        """Just a dumb presentation to show how 'gby' works.
        """
        with SQLPlus(':memory:') as conn:
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
            conn.save(load_csv("data/iris.csv", header="no,sl,sw,pl,pw,sp"), name="iris")
            a = list(conn.run("select * from iris order by sl"))
            b = list(gflat(gby(conn.run("select * from iris order by sl"), "sl")))
            for a1, b1 in zip(a, b):
                self.assertEqual(a1.sl, b1.sl)
                self.assertEqual(a1.pl, b1.pl)

    def test_run_over_run(self):
        with SQLPlus(':memory:') as conn:
            conn.save(load_csv("data/iris.csv", header="no,sl,sw,pl,pw,sp"), name="iris1")
            conn.save(load_csv("data/iris.csv", header="no,sl,sw,pl,pw,sp"), name="iris2")
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

    def test_del(self):
        """tests column deletion
        """
        with SQLPlus(':memory:') as conn:
            conn.save(load_csv('data/co2.csv'), name='co2')
            def co2_less(*col):
                """remove columns"""
                co2 = conn.run("select * from co2")
                for r in co2:
                    for c in col:
                        delattr(r, c)
                    yield r

            print('\nco2 table')
            print('==============================================================')
            conn.show("select * from co2", n=5)
            print("\nco2 table without plant and number column")
            print("order of columns not preserved")
            print('==============================================================')
            # of course you can call conn.show(co2_less('plant'), n=5)
            conn.show(co2_less, args=('plant', 'no'), n=5)
            print('==============================================================')

            conn.save(co2_less, args=('plant', 'no'))

            self.assertEqual(len(conn.table_info('co2').columns), 6)
            self.assertEqual(len(conn.table_info('co2_less').columns), 4)

    def test_column_name_validity(self):
        with SQLPlus(':memory:') as conn:
            with self.assertRaises(AssertionError):
                # sl.size is not a valid column name
                next(load_csv('data/iris.csv', header="no sl.size sw pl pw species"))

            with self.assertRaises(AssertionError):
                # 'no' is not a valid column name
                next(load_csv('data/iris.csv', header="'no' sl sw pl pw species"))

            # no_1 should be ok, underscore is ok as far as it's not the first char
            try:
                next(load_csv('data/iris.csv', header="no_1 sl sw pl pw species"))
            except:
                self.fail("no_1 is not a valid column name")

    def test_saving_csv(self):
        import os
        with SQLPlus(':memory:') as conn:
            iris = load_csv('data/iris.csv', header="no sl sw pl pw sp")
            conn.show(gby(iris, "sp"), n=2, filename='sample.csv')
            # each group contains 50 rows, hence 100
            self.assertEqual(len(list(load_csv('sample.csv'))), 100)
            os.remove('sample.csv')

unittest.main()
