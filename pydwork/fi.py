"""
Frequently used financial data work
"""

import statistics as st

from itertools import  takewhile, dropwhile, product
from collections import OrderedDict

from scipy.stats import ttest_1samp

from .sqlplus import Rows, Row, Box
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

        for rs1 in self.num(col).order(self.dcol).group(self.dcol):
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

        for rs1 in self.num(pncols + [col]).order(self.dcol).group(self.dcol):
            for rs2 in rs1.order(pncols + [col]).group(pncols):
                for pn, rs3 in enumerate(nchunks(rs2, n), 1):
                    for r in rs3:
                        r[pncol] = pn

        self.pncols[pncol] = n
        return self

    def pavg(self, col, wcol=None, pncols=None):
        "portfolio average,  wcol: weight column"

        pncols = listify(pncols) if pncols else list(self.pncols)
        ns = [self.pncols[pncol] for pncol in pncols]

        newrs = self.num(pncols + [col, wcol]) if wcol \
                else self.num(pncols + [col])

        result = []
        for rs1 in newrs.group(self.dcol):
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

    def pat(self, pncols=None):
        "average pattern, returns a box"
        pncols = listify(pncols) if pncols else list(self.pncols)

        if len(pncols) == 1:
            return self._pat1(*pncols)
        elif len(pncols) == 2:
            return self._pat2(*pncols)
        else:
            return self._patn(pncols)

    def _pat1(self, pncol):
        result = []
        for rs in self.order([pncol, self.dcol]).group(pncol):
            result.append([r[self.acol] for r in rs])

        line = []
        line.append(self.acol)
        for seq in result:
            line.append(aseq(seq))
        seq = [h - l for h, l in zip(result[-1], result[0])]
        line.append(aseq(seq, True))
        return Box([line])

    def _pat2(self, pncol1, pncol2):
        result = []
        for rs1 in self.order([pncol1, pncol2, self.dcol]).group(pncol1):
            result1 = []
            for rs2 in rs1.group(pncol2):
                result1.append([r[self.acol] for r in rs2])
            result.append(result1)

        line = []
        line.append("%s\\%s" % (pncol1[3:], pncol2[3:]))
        for i in range(1, len(result[0]) + 1):
            line.append(i)
        line.append('P%s - P1' % (i, ))

        lines = []
        lines.append(line)
        for i, result1 in enumerate(result, 1):
            line = []
            line.append(i)
            for seq in result1:
                line.append(aseq(seq))
            seq = [h - l for h, l in zip(result1[-1], result1[0])]
            line.append(aseq(seq, True))
            lines.append(line)

        line = []
        line.append("P%s - P1" % (i,))
        for hseq, lseq in zip(result[-1], result[0]):
            seq = [h - l for h, l in zip(hseq, lseq)]
            line.append(aseq(seq, True))
        lines.append(line)

        return Box(lines)

    def _patn(self, pncols):
        lines = []
        for rs in self.order(pncols[:-2]).group(pncols[:-2]):
            line = []
            for pncol in pncols[:-2]:
                line.append(pncol)
                line.append(rs[0][pncol])
            lines.append(line)

            prows = PRows(rs, self.dcol)
            prows.acol = self.acol
            lines += prows.pat(pncols[-2:]).lines
        return Box(lines)

    def tsavg(self, cols=None):
        "show time series average"
        cols = listify(cols) if cols else self[0].columns
        lines = []
        lines.append(cols)
        lines.append([aseq(self[col], True) for col in cols])
        return Box(lines)

    def famac(self, model):
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
