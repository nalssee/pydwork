"""
SQLite3 based utils for statistical analysis

Emperical data analysis task is largely composed of two parts,
wrangling data(cleaning up so you can easily handle it)
and applying statistical methodologies to data.

Python is really great for data analysis, since you have
widely used, reliable tools like Pandas, Numpy, Scipy, ...

Pandas is said-to-be a really great for wrangling data.
But it wasn't a very smooth process for me to learn Pandas.
And you have to load all the data in the system memory.
So you have to figure out a way around when you need to
handle very large data sets.

If you focus only on data wragling, SQL is an amazingly great
tool. Performance, flexibility, reliablity, ease of use, it will
never disappoint you.

But there's one problem in SQL for data analysis. It doesn't
integrate with python functions out of box. And it doesn't provide
sophisticated statistical functions.

Say you want to group stock returns every year
,do some linear regressions with variables of your interests,
and sum up the coefficients from the task.
If you have to do the work using plain SQL,
your life won't be happy anymore.

It would be great if you can integrate Python with SQL.
It is what I'm trying to do here.

This program is not a sophisticated, full-fledged automation system.
I'd rather say it's just a mental framework.

Use SQL to clean up data and group them by certain criteria,
apply statical tools to each group using Python(statmodels, numpy and so on)
and sum up the results.

If you have some basic SQL and Python knowledge,
(You don't have to be an expert)
I believe this is better that using Pandas for some people.
For me, it is.


As for docstring:
    GF[int->int]: generator function type that yields int
                  and takes int as arg, parameters are optional
    FN[str->str]: Ordinary function type
    iter: types.GeneratorType
    Iter[str]: iterator of strings
    file: file object

"""

import os
import sys
import csv
import fileinput
import re
import sqlite3
import tempfile

from collections import Counter
from contextlib import contextmanager
from functools import wraps
from itertools import chain, groupby, islice

from bs4 import BeautifulSoup

import pandas as pd

__all__ = ['dbopen', 'Row', 'gby', 'reel',
           'reel_html_table', 'pick',
           'prepend_header', 'adjoin', 'disjoin',
           'todf', 'torows',
           'set_workspace', 'get_workspace']


WORKSPACE = ''


class Row:
    """
    Basically the same as sqlite3.Row
    it's just that using sqlite3.Row is a bit clunkier.
    r['col'] = 34
    I prefer r.col = 34

    And it's better to keep the order of columns
    """

    def __init__(self):
        super().__setattr__('_columns', [])
        super().__setattr__('_values', [])

    @property
    def columns(self):
        "List[str]: column names"
        return self._columns

    @property
    def values(self):
        "List[type]"
        return self._values

    def __setattr__(self, name, value):
        if name not in self.__dict__:
            self.columns.append(name)
            self.values.append(value)
        else:
            # Updating the attribute value of a row
            idx = self.columns.index(name)
            self.values[idx] = value
        super().__setattr__(name, value)

    def __delattr__(self, name):
        idx = self.columns.index(name)
        del self.columns[idx]
        del self.values[idx]
        super().__delattr__(name)

    def __str__(self):
        return str(list(zip(self.columns, self.values)))


class SQLPlus:
    """
    Attributes:
        tables (List[str]): list of all tables in the DB
    """

    def __init__(self, dbfile):
        """
        Args:
            dbfile (str): db filename or ':memory:'
        """
        # self._dbfile = os.path.join(WORKSPACE, dbfile)
        self._dbfile = dbfile if dbfile == ':memory:' \
            else os.path.join(WORKSPACE, dbfile)
        self.conn = sqlite3.connect(self._dbfile)
        self._cursor = self.conn.cursor()
        self.tables = self._list_tables()

    # args can be a list, a tuple or a dictionary
    def run(self, query, args=()):
        """Simply executes sql statement and update tables attribute

        Args:
            query (str): SQL query string
            args (List[any] or Tuple[any]): args for SQL query
        """
        query = query.lower()
        self._cursor.execute(query, args)

        self.tables = self._list_tables()

    def reel(self, query, args=()):
        """Generates a sequence of rows from a query.

        Args:
            query (str): select statement or table name

        Yields:
            Row
        """
        query = _select_statement(query.lower())
        if query.strip().partition(' ')[0].upper() != "SELECT":
            raise ValueError("use 'run' for ", query)
        qrows = self._cursor.execute(query, args)
        columns = [c[0] for c in qrows.description]

        # there can't be duplicates in column names
        if len(columns) != len(set(columns)):
            raise ValueError('duplicates in columns names')

        for qrow in qrows:
            row = Row()
            for col, val in zip(columns, qrow):
                setattr(row, col, val)
            yield row

    def save(self, seq, name=None, args=(), n=None):
        """create a table from an iterator.

        Note:
            if seq is a generator function and 'name' is not given,
            the function name is going to be the table name.

        Args:
            seq (iter or GF[* -> Row])
            name (str): table name in DB
            args (List[type]): args for seq (GF)
            n (int): number of rows to save
        """
        def save_rows_to_tempfile(f, rs):
            """
            Args:
                f (file)
                rs (Iter[Row])
            """
            def tval(val):
                """If val contains a comma or newline it causes problems
                So just remove them.
                There might be some other safer methods but I don't think
                newlines or commas are going to affect any data analysis.

                Args:
                    val (type)
                Returns:
                    str
                """
                return str(val).replace(',', ' ').replace('\n', ' ')

            for r in rs:
                vals = [tval(v) for v in r.values]
                f.write((','.join(vals) + '\n').encode())

        # single letter variable is hard to find
        nrows = n

        # if 'seq' is a generator function, it is executed to make an iterator
        if hasattr(seq, '__call__'):
            name = name or seq.__name__
            seq = seq(*args)

        row0, seq = _peek_first(seq)

        if nrows:
            seq = islice(seq, nrows)

        if name in self.tables:
            return
        if name is None:
            raise ValueError('table name required')

        colnames = row0.columns

        # You can't save the iterator directly because
        # once you execute a table creation query,
        # then the query becomes the most recent query,
        # not the query for the iterator.
        # Which means, if you iterate over the iterator,
        # it iterates over the table creation query.

        # You can see the example at test/sqlplus_test.py
        # 'test_run_over_run'

        # So you save the iterator up in a temporary file
        # and then save the file to a database.
        # In the process column types are checked.

        # This is the root of all evil
        with tempfile.TemporaryFile() as fport:
            # Write the iterator in a temporary file
            # encode it as binary.
            save_rows_to_tempfile(fport, seq)

            # create table
            istmt = _insert_statement(name, len(colnames))
            self._cursor.execute(_create_statement(name, colnames))

            # Now insertion to a DB
            fport.seek(0)

            for line in fport:
                # line[:-1] because last index indicates '\n'
                line_vals = line[:-1].decode().split(',')
                self._cursor.execute(istmt, line_vals)

        self.tables = self._list_tables()

    # Be careful so that you don't overwrite the file
    def show(self, query, args=(), n=30, cols=None,
             filename=None, overwrite=True):
        """Printing to a screen or saving to a file

        Args:
            query (str or Iter[Row] or GF)
            args (List[type] or Tuple[type]): args for query (GF)
            n (int): maximum number of lines to show
            cols (str or List[str]): columns to show
            filename (str): filename to save
            overwrite (bool): if true overwrite a file
        """
        # so that you can easily maintain code
        # Searching nrows is easier than searching n in editors
        nrows = n

        if isinstance(query, str):
            seq_rvals = self._cursor.execute(
                _select_statement(query, cols or '*'), args)
            colnames = [c[0] for c in seq_rvals.description]

        # then query is an iterator of rows, or a list of rows
        # of course it can be just a generator function of rows
        else:
            rows = query
            if hasattr(rows, '__call__'):
                rows = rows(*args)

            if cols:
                rows = pick(cols, rows)

            row0, rows = _peek_first(rows)

            colnames = row0.columns
            seq_rvals = (r.values for r in rows)

        # write to a file
        if filename:
            # practically infinite number
            nrows = nrows or sys.maxsize

            if not filename.endswith('.csv'):
                filename = filename + '.csv'

            if os.path.isfile(os.path.join(WORKSPACE, filename)) \
               and not overwrite:
                return

            with open(os.path.join(WORKSPACE, filename), 'w') as fout:
                fout.write(','.join(colnames) + '\n')
                for rvals in islice(seq_rvals, nrows):
                    fout.write(','.join([str(val) for val in rvals]) +
                               '\n')
        # write to stdout
        else:
            # show practically all rows, columns.
            with pd.option_context("display.max_rows", nrows), \
                    pd.option_context("display.max_columns", 1000):
                # make use of pandas DataFrame displaying
                seq_rvals_list = list(islice(seq_rvals, nrows + 1))
                print(pd.DataFrame(seq_rvals_list[:nrows],
                                   columns=colnames))
                if len(seq_rvals_list) > nrows:
                    print("...more rows...")

    # Simpler version of show (when you write it to a file)
    # so you make less mistakes.
    def write(self, query, filename=None, args=()):
        """
        Args:
            query (str or Iter[Row] or GF)
            args (List[type] or Tuple[type]): args for query (GF)
            filename (str): filename to save
        """
        if isinstance(query, str) and \
           _is_oneword(query) and filename is None:
            filename = query
        self.show(query, filename=filename, args=args, n=None, overwrite=True)

    def _list_tables(self):
        """List of table names in the database

        Returns:
            List[str]
        """
        query = self._cursor.execute("""
        select * from sqlite_master
        where type='table'
        """)
        tables = [row[1].lower() for row in query]
        return sorted(tables)

    def summarize(self, n=1000, overwrite=True):
        """
        Args:
            n (int)
            overwrite (bool)
        """
        summary_dir = os.path.join(WORKSPACE, 'summary')
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

        for table in self.tables:
            filename = os.path.join(summary_dir, table + '.csv')
            self.show(table, n=n, filename=filename, overwrite=overwrite)

    def drop(self, table):
        """
        drop table if exists

        Args:
            table (str)
        """
        # you can't use '?' for table name
        # '?' is for data insertion
        self.run('drop table if exists %s' % (table,))
        summary_dir = os.path.join(WORKSPACE, 'summary')
        filename = os.path.join(summary_dir, table + '.csv')
        if os.path.isfile(filename):
            # remove summary file as well if exists
            os.remove(filename)

        self.tables = self._list_tables()

    def count(self, seq):
        """
        count the size of a sequence

        Args:
            seq (str or GF or iter)
        Returns:
            int
        """
        if isinstance(seq, str):
            seq = self._cursor.execute(_select_statement(seq))
        if hasattr(seq, '__call__'):
            seq = seq()
        return sum(1 for _ in seq)


@contextmanager
def dbopen(dbfile):
    """Connects to SQL database(sqlite)

    Args:
        dbfile (str)
    Yields:
        SQLPlus
    """
    splus = SQLPlus(dbfile)
    try:
        yield splus
    finally:
        splus.conn.commit()
        splus.conn.close()


def gby(seq, key):
    """Group the iterator by a key

    Args
        seq (iter)
        key (FN[Row -> type] or List[str] or str): if [], group them all
    Yields:
        List[Row]
    """
    if not hasattr(key, '__call__'):
        key = _build_keyfn(key)
    for _, rows in groupby(seq, key):
        # to list or not to list
        yield list(rows)


def todf(rows):
    """
    Args:
        List[Row]
    Returns:
        pd.DataFrame
    """
    colnames = rows[0].columns
    d = {}
    for col in zip(colnames, *(r.values for r in rows)):
        d[col[0]] = col[1:]
    return pd.DataFrame(d)


# This is not an ordinary function!!
# So, x != torows(todf(x))
# efficiency issue
# Most of the time, in fact almost always,
# you will use this function to use with yield from
# so there's no point in returning a list of rows
def torows(df):
    """
    Args:
        df (pd.DataFrame)
    Yields:
        Row
    """
    colnames = df.columns.values
    for vals in df.values.tolist():
        r = Row()
        for c, v in zip(colnames, vals):
            setattr(r, c, v)
        yield r


def pick(cols, seq):
    """
    Args:
        cols (str or List[str])
        seq (Iter[Row])
    Yields:
        Row
    """
    cols = _listify(cols)
    for r in seq:
        r1 = Row()
        for c in cols:
            setattr(r1, c, getattr(r, c))
        yield r1


def prepend_header(filename, header=None, drop=1):
    """Drop n lines and prepend header

    Args:
        filename (str)
        header (str)
        drop (int)
    """
    for no, line in enumerate(
            fileinput.input(os.path.join(WORKSPACE, filename), inplace=True)):
        # it's meaningless to set drop to -1, -2, ...
        if no == 0 and drop == 0:
            if header:
                print(header)
            print(line, end='')
        # replace
        elif no + 1 == drop:
            if header:
                print(header)
        elif no >= drop:
            print(line, end='')
        else:
            # no + 1 < drop
            continue


def convtype(val):
    """Convert type if possible

    Args:
        val (str)
    Returns:
        int or float or str
    """
    try:
        return int(val)
    except:
        try:
            return float(val)
        except:
            return val


def reel(csv_file, header=None):
    """Loads well-formed csv file, 1 header line and the rest is data

    Args:
        csv_file (str)
        header (str)
    Yields:
        Row
    """
    def is_empty_line(line):
        """Tests if a list of strings is empty for example ["", ""] or []
        """
        return [x for x in line if x.strip() != ""] == []

    if not csv_file.endswith('.csv'):
        csv_file += '.csv'
    with open(os.path.join(WORKSPACE, csv_file)) as fin:
        first_line = fin.readline()[:-1]
        header = header or first_line
        columns = _gen_valid_column_names(_listify(header))
        ncol = len(columns)
        for line_no, line in enumerate(csv.reader(fin)):
            if len(line) != ncol:
                if is_empty_line(line):
                    continue
                # You've read a line alread, so line_no + 1
                raise ValueError("%s at %s invalid line" %
                                 (csv_file, line_no + 1))
            row1 = Row()
            for col, val in zip(columns, line):
                setattr(row1, col, convtype(val))
            yield row1


# Inflexible, experimental
def reel_html_table(html_file, css_selector='table'):
    """Read a simple well formed table

    Args:
        html_file (str)
        css_selector (str)
    Yields:
        Row
    """
    if not html_file.endswith('.html'):
        html_file += '.html'
    with open(os.path.join(WORKSPACE, html_file)) as fin:
        soup = BeautifulSoup(fin, 'html.parser')
        trs = soup.select(css_selector)[0].select('tr')
        colnames = _gen_valid_column_names(
            [x.text for x in trs[0].select('th')])
        for tr in trs[1:]:
            r = Row()
            vals = [x.text for x in tr.select('td')]
            for col, val in zip(colnames, vals):
                setattr(r, col, convtype(val))
            yield r


def adjoin(colnames):
    """Decorator to ensure that the rows to have the columns for sure

    Args:
        colnames (str or List[str])
    Returns:
        FN
    """
    colnames = _listify(colnames)

    def dec(gen):
        "real decorator"
        @wraps(gen)
        def wrapper(*args, **kwargs):
            "if a column doesn't exist, append it"
            for row in gen(*args, **kwargs):
                for col in colnames:
                    try:
                        # rearrange the order
                        val = getattr(row, col)
                        delattr(row, col)
                        setattr(row, col, val)
                    except:
                        setattr(row, col, '')
                yield row
        return wrapper
    return dec


def disjoin(colnames):
    """Decorator to ensure that the rows NOT to have the columns for sure.

    Args:
        colnames (str or List[str])
    Returns:
        FN
    """
    colnames = _listify(colnames)

    def dec(gen):
        "real decorator"
        @wraps(gen)
        def wrapper(*args, **kwargs):
            "Delete a column"
            for row in gen(*args, **kwargs):
                for col in colnames:
                    # whatever it is, just delete it
                    try:
                        delattr(row, col)
                    except:
                        pass
                yield row
        return wrapper
    return dec


# Should I just export WORKSPACE variable directly?
def set_workspace(dir):
    """
    Args:
        dir (str)
    """
    global WORKSPACE
    WORKSPACE = dir


def get_workspace():
    """
    Returns:
        str
    """
    return WORKSPACE


def camel2snake(name):
    """
    Args:
        name (str): camelCase
    Returns:
        str: snake_case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _build_keyfn(key):
    """
    Args:
        key (str or List[str]): column names
    Returns:
        FN[Row -> type]
    """
    colnames = _listify(key)
    if len(colnames) == 1:
        return lambda r: getattr(r, colnames[0])
    else:
        return lambda r: [getattr(r, colname) for colname in colnames]


def _gen_valid_column_names(columns):
    """Generate valid column names from arbitrary ones

    Note:
        Every column name is lowercased
    Args:
        columns (List[str])
    Returns:
        List[str]
    Example:
        >>> _gen_valid_column_names(['a', '_b', 'a', 'a1"*c', 'a1c'])
        ['a0', 'a_b', 'a1', 'a1c0', 'a1c1']

    """
    # Some of the sqlite keywords are not allowed for column names
    # http://www.sqlite.org/sessions/lang_keywords.html
    SQLITE_KEYWORDS = {
        "ABORT", "ACTION", "ADD", "AFTER", "ALL", "ALTER", "ANALYZE", "AND",
        "AS", "ASC", "ATTACH", "AUTOINCREMENT", "BEFORE", "BEGIN", "BETWEEN",
        "BY", "CASCADE", "CASE", "CAST", "CHECK", "COLLATE", "COLUMN",
        "COMMIT", "CONFLICT", "CONSTRAINT", "CREATE", "CROSS", "CURRENT_DATE",
        "CURRENT_TIME", "CURRENT_TIMESTAMP", "DATABASE", "DEFAULT",
        "DEFERRABLE", "DEFERRED", "DELETE", "DESC", "DETACH", "DISTINCT",
        "DROP", "EACH", "ELSE",
        "END", "ESCAPE", "EXCEPT", "EXCLUSIVE", "EXISTS", "EXPLAIN", "FAIL",
        "FOR", "FOREIGN", "FROM", "FULL", "GLOB", "GROUP", "HAVING", "IF",
        "IGNORE", "IMMEDIATE", "IN", "INDEX", "INDEXED", "INITIALLY", "INNER",
        "INSERT", "INSTEAD", "INTERSECT", "INTO", "IS", "ISNULL", "JOIN",
        "KEY", "LEFT", "LIKE", "LIMIT", "MATCH", "NATURAL",
        # no is ok somehow
        # no idea why
        # "NO",
        "NOT", "NOTNULL", "NULL", "OF", "OFFSET", "ON", "OR", "ORDER", "OUTER",
        "PLAN", "PRAGMA", "PRIMARY", "QUERY", "RAISE", "REFERENCES",
        "REGEXP", "REINDEX", "RENAME", "REPLACE", "RESTRICT", "RIGHT",
        "ROLLBACK", "ROW", "SAVEPOINT", "SELECT", "SET", "TABLE", "TEMP",
        "TEMPORARY", "THEN", "TO", "TRANSACTION",
        "TRIGGER", "UNION", "UNIQUE", "UPDATE", "USING", "VACUUM", "VALUES",
        "VIEW", "VIRTUAL", "WHEN", "WHERE",

        # These are not sqlite keywords but attribute names of Row class
        'COLUMNS', 'VALUES',
    }

    DEFAULT_COLUMN_NAME = 'col'
    temp_columns = []
    for col in columns:
        # save only alphanumeric and underscore
        # and remove all the others
        newcol = camel2snake(re.sub(r'[^\w]+', '', col))
        if newcol == '':
            newcol = DEFAULT_COLUMN_NAME
        elif not newcol[0].isalpha() or newcol.upper() in SQLITE_KEYWORDS:
            newcol = 'a_' + newcol
        temp_columns.append(newcol)

    # no duplicates
    if len(temp_columns) == len(set(temp_columns)):
        return temp_columns

    # Tag numbers to column-names starting from 0 if there are duplicates
    cnt = {col: n for col, n in Counter(temp_columns).items() if n > 1}
    cnt_copy = dict(cnt)

    result_columns = []
    for col in temp_columns:
        if col in cnt:
            result_columns.append(col + str(cnt_copy[col] - cnt[col]))
            cnt[col] -= 1
        else:
            result_columns.append(col)
    return result_columns


def _listify(colstr):
    """A comma or space separated string to a list of strings

    Args:
        colstr (str)
    Returns:
        List[str]

    Example:
        >>> _listify('a b c')
        ['a', 'b', 'c']

        >>> _listify('a, b, c')
        ['a', 'b', 'c']
    """
    if isinstance(colstr, str):
        if ',' in colstr:
            return [x.strip() for x in colstr.split(',')]
        else:
            return [x for x in colstr.split(' ') if x]
    else:
        return colstr


def _peek_first(seq):
    """
    Note:
        peeked first item is pushed back to the sequence
    Args:
        seq (Iter[type])
    Returns:
        Tuple(type, Iter[type])
    """
    seq = iter(seq)
    first_item = next(seq)
    return first_item, chain([first_item], seq)


def _create_statement(name, colnames):
    """create table if not exists foo (...)

    Note:
        Every type is numeric.
        Table name and column names are all lower cased
    Args:
        name (str): table name
        colnames (List[str])
    Returns:
        str
    """
    schema = ', '.join([col.lower() + ' ' + 'numeric' for col in colnames])
    return "create table if not exists %s (%s)" % (name.lower(), schema)


def _insert_statement(name, ncol):
    """insert into foo values (?, ?, ?, ...)

    Note:
        Column name is lower cased
    Args:
        name (str): table name
        ncol (int)
    Returns:
        str
    """
    qmarks = ', '.join(['?'] * ncol)
    return "insert into %s values (%s)" % (name.lower(), qmarks)


def _is_oneword(query):
    """
    Args:
        query (str)
    Returns:
        bool
    """
    return len(query.strip().split(' ')) == 1


def _select_statement(query, cols='*'):
    """If query is just one word, turn it to a select stmt
    or just leave it

    Args:
        query (str)
        cols (str or List[str])
    Returns:
        str
    """
    if _is_oneword(query):
        return "select %s from %s" % (', '.join(_listify(cols)), query)
    return query
