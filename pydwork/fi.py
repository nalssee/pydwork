"""
Frequently used financial data work
"""

import statistics as st

from itertools import  takewhile, dropwhile, product
from collections import OrderedDict

from scipy.stats import ttest_1samp

from .sqlplus import Rows, Row
from .util import nchunks, listify, yyyymm, yyyymmdd


class PRows(Rows):
    """Rows for portfolio analysis
    """
    def __init__(self, rows, dcol):
        Rows.__init__(self, rows)
        # date column
        self.dcol = dcol
        # (computed) average column
        self.acol = None
        # portfolio number columns
        self.pncols = OrderedDict()

    def pn(self, col, n):
        "portfolio numbering for independent sort"
        pncol = 'pn_' + col
        for r in self:
            r[pncol] = ''

        self.num(col)
        for rs1 in self.order(self.dcol).group(self.dcol):
            for pn, rs2 in enumerate(nchunks(rs1.order(col), n), 1):
                for r in rs2:
                    r[pncol] = pn

        self.pncols[pncol] = n
        return self

    def dpn(self, col, n, pncols=None):
        """
        portfolio numbering for dependent sort
        if you don't specify pncols, self.pncols is used
        """
        pncol = 'pn_' + col

        for r in self:
            r[pncol] = ''

        pncols = listify(pncols) if pncols else list(self.pncols)

        self.num(pncols + [col])
        for rs1 in self.order(self.dcol).group(self.dcol):
            for rs2 in rs1.order(pncols + [col]).group(pncols):
                for pn, rs3 in enumerate(nchunks(rs2, n), 1):
                    for r in rs3:
                        r[pncol] = pn

        self.pncols[pncol] = n
        return self

    def avg(self, col, wcol=None, pncols=None):
        "wcol: weight column"
        self.num([col] + [wcol]) if wcol else self.num(col)

        pncols = listify(pncols) if pncols else list(self.pncols)
        ns = [self.pncols[pncol] for pncol in pncols]

        result = []
        for rs1 in self.group(self.dcol):
            for pns, rs2 in zip(product(*(range(1, n + 1) for n in ns)),
                                rs1.order(pncols).group(pncols)):
                # test if there's any missing portfolio
                if [rs2[0][pncol] for pncol in pncols] != list(pns):
                    raise ValueError('missing portfolio no. %s in %s' %  \
                        (list(pns), rs2[0][self.dcol]))

                r = Row()
                r[self.dcol] = rs2[0][self.dcol]
                for pncol, pn in zip(pncols, pns):
                    r[pncol] = pn
                r.n = len(rs2)
                r[col] = wavg(rs2, col, wcol) if wcol else st.mean(rs2[col])
                result.append(r)

        prows = PRows(result, self.dcol)
        prows.pncols = OrderedDict()
        for pncol, n in zip(pncols, ns):
            prows.pncols[pncol] = n
        prows.acol = col
        return prows

    def pshow(self, pncols=None):
        "show pattern"
        pncols = listify(pncols) if pncols else list(self.pncols)

        if len(pncols) == 1:
            self._pshow1(*pncols)
        elif len(pncols) == 2:
            self._pshow2(*pncols)
        else:
            self._pshowN(pncols)

    def _pshow1(self, pncol):
        result = []
        for rs in self.order([pncol, self.dcol]).group(pncol):
            result.append([r[self.acol] for r in rs])

        print(self.acol, end=',')
        for seq in result:
            print(aseq(seq), end=',')
        seq = [h - l for h, l in zip(result[-1], result[0])]
        print(aseq(seq, True))

    def _pshow2(self, pncol1, pncol2):
        result = []
        for rs1 in self.order([pncol1, pncol2, self.dcol]).group(pncol1):
            line = []
            for rs2 in rs1.group(pncol2):
                line.append([r[self.acol] for r in rs2])
            result.append(line)
        print("%s\\%s" % (pncol1[3:], pncol2[3:]), end=',')
        for i in range(1, len(result[0]) + 1):
            print(i, end=',')
        print('P%s - P1' % (i, ))
        for i, line in enumerate(result, 1):
            print(i, end=',')
            for seq in line:
                print(aseq(seq), end=',')
            seq = [h - l for h, l in zip(line[-1], line[0])]
            print(aseq(seq, True))
        print("P%s - P1" % (i,), end=',')
        for hseq, lseq in zip(result[-1], result[0]):
            seq = [h - l for h, l in zip(hseq, lseq)]
            print(aseq(seq, True), end=',')
        print()

    def _pshowN(self, pncols):
        for rs in self.order(pncols[:-2]).group(pncols[:-2]):
            for pncol in pncols[:-2]:
                print(pncol, rs[0][pncol], end=' ')
            print()
            prows = PRows(rs, self.dcol)
            prows.acol = self.acol
            prows.pshow(pncols[-2:])

    def tshow(self, cols=None):
        cols = listify(cols) if cols else self[0].columns
        print(','.join(cols))
        print(','.join(aseq(self[col], True) for col in cols))

    def famac(self, model, show=True):
        "Fama Macbeth"
        def xvars(model):
            _, right = model.split('~')
            return [x.strip() for x in right.split('+')]

        xvs = ['intercept'] + xvars(model)

        params = []
        for rs1 in self.order(self.dcol).group(self.dcol):
            reg = rs1.ols(model)
            r = Row()
            r[self.dcol] = rs1[0][self.dcol]
            for var, p in zip(xvs, reg.params):
                r[var] = p
            r.n = int(reg.nobs)
            r.r2 = reg.rsquared
            params.append(r)

        prows = PRows(params, self.dcol)
        if show:
            prows.tshow(xvs)
        return prows

    def roll(self, period, jump):
        "group rows over time, allowing overlaps"
        def get_nextdate(date, period):
            "date after the period"
            if len(date) == 8:
                return yyyymmdd(date, period)
            elif len(date) == 6:
                return yyyymm(date, period)
            elif len(date) == 4:
                return int(date) + period
            else:
                raise ValueError('Invalid date', date)

        def rows_for(rs, date, period):
            "rows from the date for the period"
            enddate = get_nextdate(date, period)
            return PRows(takewhile(lambda r: r[self.dcol] < enddate, rs),
                         self.dcol)

        rs = self.rows
        while rs != []:
            startdate = str(rs[0][self.dcol])
            enddate = get_nextdate(startdate, jump)
            yield rows_for(rs, startdate, period)
            rs = list(dropwhile(lambda r: r[self.dcol] < enddate, rs))


def aseq(seq, tval=False):
    "average of sequence"
    tstat = ttest_1samp(seq, 0)
    if tval:
        return "%s (%s)" % \
               (star(st.mean(seq), tstat[1]), round(tstat[0], 2))
    return round(st.mean(seq), 3)


def wavg(rs, col, wcol):
    "compute weigthed average"
    total = sum(r[wcol] for r in rs)
    return st.mean(r[col] * r[wcol] / total for r in rs)


def star(val, pval):
    "put stars according to p-value"
    if pval < 0.001:
        return str(round(val, 3)) + '***'
    elif pval < 0.01:
        return str(round(val, 3)) + '**'
    elif pval < 0.05:
        return str(round(val, 3)) + '*'
    else:
        return str(round(val, 3))
