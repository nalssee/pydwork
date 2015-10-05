"""Table element from bs4 => DataFrame in pandas"""


import bs4
import pandas as pd
from itertools import zip_longest

import unittest


__all__ = ['soup2df', 'HTBLError', 'InvalidSpanAttr', 'InvalidTable']


class HTBLError(Exception):
    """Root exception for this module. """
    pass


class InvalidSpanAttr(HTBLError):
    """rowspan or colspan attributute values are not valid for int """
    pass


class InvalidTable(HTBLError):
    """Table size is ridiculous"""
    pass


def soup2rows(soup):
    """table element soup to a list of lists with 'td's and 'th's"""
    rows = []
    for row in soup.find_all('tr'):
        rows.append([elt for elt in row.find_all(['td', 'th'])])
    return rows


def expand_span(soup):
    """colspan and rowspan attributes must be handled.
    For example, if colspan is 3 then it expands to 3 elements

    Most of table validity test is performed here. """

    def fillin(elt, i, j, expanded_rows):
        """Fill in elt in the first non-None cell in expanded_rows """
        rowspan, colspan = get_span(elt)
        # Find the index of first non 'None' element
        try:
            newj = expanded_rows[i].index(None)
        except ValueError as e:
            raise InvalidTable(soup) from e

        for i1 in range(rowspan):
            for j1 in range(colspan):
                try:
                    expanded_rows[i + i1][newj + j1] = elt
                except IndexError as e:
                    raise InvalidTable(soup) from e

    rows = soup2rows(soup)

    height = len(rows)
    width = sum([get_span(elt)[1] for elt in rows[0]])

    expanded_rows = [[None for _ in range(width)]
                     for _ in range(height)]
    for i, row in enumerate(rows):
        for j, elt in enumerate(row):
            fillin(elt, i, j, expanded_rows)
    # Tests if unfilled cell exists
    if any(None in row for row in expanded_rows):
        raise InvalidTable(soup)

    return expanded_rows


def get_span(elt):
    """Returns a tuple of rowspan and colspan

    1 if no such attribute exist"""
    try:
        rowspan = 1 if not elt.has_attr('rowspan') else int(elt['rowspan'])
        colspan = 1 if not elt.has_attr('colspan') else int(elt['colspan'])
        return rowspan, colspan
    except ValueError as e:
        raise InvalidSpanAttr(elt) from e


def soup2df(soup):
    """Returns a dataframe that keys are heads and values are
    just lists of corresponding body part of rows

    key type: string, value type: list of strings
    (No type conversion)

    head part is collapsed if there are multiple heads.
    body part is all the other rows that are not heads

    In search of heads, first filter any rows that's composed of
    only 'th' elements, if there are none, do the same for transposed rows

    ***** CAUTION!! *****
    If heads are in the rows and also columns, it is assumed to be a
    rows-based table. """

    def head_body1(rows):
        head_rows, body_rows = [], []
        for row in rows:
            if all(elt.name == 'th' for elt in row):
                head_rows.append(row)
            else:
                body_rows.append(row)
        return head_rows, body_rows

    def expand_br(rows):
        def expand_br1(row, n):
            result = []
            for elt in row:
                xs = elt.split('<br>')
                result.append(xs + ([''] * (n - len(xs))))
            return [r for r in tr(result) if ''.join(r).strip() != '']
        result = []
        for row in rows:
            n = max(elt.count('<br>') for elt in row)
            if n == 0:
                result.append(row)
            else:
                result.extend(expand_br1(row, n + 1))
        return result

    rows = expand_span(soup)

    head, body = head_body1(rows)
    if head == []:
        head, body = head_body1(tr(rows))

    # If head == [] after all this, some random column names generated
    if head == []: body = tr(body)

    columns = collapse_head(head) \
              if head != [] else ['NONAME' + str(i) for i in range(len(body[0]))]

    data = expand_br(map2d(get_text_td, body))

    return pd.DataFrame(columns=columns, data=data)


def tr(xs):
    """transpose"""
    return [list(elt) for elt in zip(*xs)]


def collapse_head(head_rows, delim='\n'):
    """First extract values from each element and clean it up.
    If head is composed of multiple lines join them with '\n'

    If there are duplicates, append counters at the end of each string."""
    head_names = [delim.join(names if len(set(names)) != 1 else names[:1])
                  for names in tr(map2d(get_text_th, head_rows))]

    # It is posssible that some neighboring names are the same
    # owing to 'colspan' or 'rowspan' attributes
    result = []
    counter = 1
    for h1, h2 in zip_longest(head_names, head_names[1:]):
        if h1 == h2:
            result.append(h1 + str(counter))
            counter += 1
        else:
            if counter == 1:
                result.append(h1)
            else:
                result.append(h1 + str(counter))
                counter = 1
    return result


# I can't find a built-in function for this
def map2d(fn, xxs):
    """Apply fn to a list of lists"""
    return [[fn(x) for x in xs] for xs in xxs]


def get_text_th(elt):
    """None-alphanumeric characters are replaced with underscores

    If the first charater is a digit 'A' is appended.
    If it's an empty string 'NONAME' is returned """
    result = "".join([c if c.isalnum() else '_'
                      for c in elt.get_text().strip()])
    if result == "":
        return "NONAME"
    elif result[0].isdigit():
        return 'A' + result
    else:
        return result


def get_text_td(elt):
    """ If comma-removed string can be converted to a number
    then comma is replaced with ''.

    Some of those elts with other tags(mostly <br>) inside,
    those tags are converted with a string '<br>'.

    Otherwise simple comma is relaced with a blank"""
    text = ''.join([x if isinstance(x, str) else \
                    '<br>' for x in list(elt.contents)]).strip()
    try:
        newtext = text.replace(',', '').replace('(-)', '-')
        float(newtext)
        return newtext
    except ValueError:
        # We are going to turn it into a csv (mostly)
        # So avoiding commas
        return text.replace(',', ' ')


# ===================================================
# TEST!!!
# The following table sampels are scrapted from "http://w3school.com"
# ===================================================

# Simple one
TBL1 = """
<!DOCTYPE html>
<html>
  <body>
    <table style="width:100%">
      <tr>
	<th>First Name</th>
	<th>Last Name</th>
	<th>Points</th>
      </tr>
      <tr>
	<td>Jill</td>
	<td>Smith</td>
	<td>50</td>
      </tr>
      <tr>
	<td>Eve</td>
	<td>Jackson</td>
	<td>94</td>
      </tr>
      <tr>
	<td>John</td>
	<td>Doe</td>
	<td>80</td>
      </tr>
    </table>
  </body>
</html>
"""


# colspan
TBL2 = """
<table style="width:100%">
  <tr>
    <th>Name</th>
    <th colspan="2">Telephone</th>
  </tr>
  <tr>
    <td>Bill Gates</td>
    <td>555 77 854</td>
    <td>555 77 855</td>
  </tr>
</table>
"""


# rowspan
TBL3 = """
<!DOCTYPE html>
<html>
  <body>
    <h2>Cell that spans two rows:</h2>
    <table style="width:100%">
      <tr>
	<th>Name:</th>
	<td>Bill Gates</td>
      </tr>
      <tr>
	<th rowspan="2">Telephone:</th>
	<td>555 77 854</td>
      </tr>
      <tr>
	<td>555 77 855</td>
      </tr>
    </table>
  </body>
</html>
"""


# <br>
TBL4 = """
<table style="width:100%">
  <tr>
    <th>Name</th>
    <th colspan="2">Telephone</th>
  </tr>
  <tr>
    <td>Bill Gates</td>
    <td>555<br>\"<br>854</td>
    <td>555<br>\"<br>855</td>
  </tr>
</table>
"""

class HTBLTest(unittest.TestCase):
    tbl1, tbl2, tbl3, tbl4 = [bs4.BeautifulSoup(tbl).find('table')
                              for tbl in [TBL1, TBL2, TBL3, TBL4]]
    def test_soup2rows(self):
        rows1, rows2, rows3 = [soup2rows(tbl) \
                               for tbl in [self.tbl1, self.tbl2, self.tbl3]]
        self.assertEqual(len(rows1), 4)
        self.assertEqual(len(rows2), 2)
        self.assertEqual(len(rows3), 3)
        # columns check
        self.assertEqual(len(rows1[0]), 3)

        self.assertEqual(len(rows2[0]), 2)
        self.assertEqual(len(rows2[1]), 3)

        self.assertEqual(len(rows3[0]), 2)
        self.assertEqual(len(rows3[1]), 2)
        self.assertEqual(len(rows3[2]), 1)

    def test_expand_span(self):
        # as for a simple table like tbl1,
        # expand_span doesn't do anything
        self.assertEqual(soup2rows(self.tbl1), expand_span(self.tbl1))

        rows2 = expand_span(self.tbl2)
        self.assertEqual(len(rows2[0]), 3)
        self.assertEqual(len(rows2[1]), 3)

        rows3 = expand_span(self.tbl3)
        self.assertEqual(len(rows3[0]), 2)
        self.assertEqual(len(rows3[1]), 2)
        self.assertEqual(len(rows3[2]), 2)

    # Havn't written yet, see for yourself
    def test_soup2df(self):
        print(soup2df(self.tbl1), end='\n\n')
        print(soup2df(self.tbl2), end='\n\n')
        print(soup2df(self.tbl3), end='\n\n')
        print(soup2df(self.tbl4), end='\n\n')


if __name__ == '__main__':
    unittest.main()
