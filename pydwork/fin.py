"""
Frequently used financial data work
"""

import statistics as st

from itertools import  takewhile, dropwhile, product
from collections import OrderedDict

from scipy.stats import ttest_1samp

from .sqlplus import Rows, Row, Box
from .util import nchunks, listify, grouper, yyyymm, yyyymmdd, breaks, isnum


class PRows(Rows):
    """Rows for portfolio analysis
    """
    def __init__(self, rows, dcol, fcol=None):
        Rows.__init__(self, rows)
        # date column
        self.dcol = dcol
        # firm id column
        self.fcol = fcol
        # (computed) average column
        self.acol = None
        # portfolio number columns
        self.pncols = OrderedDict()

    def reset(self):
        "Remove pncols and initiate acol and pncols"
        pncols = [col for col in self.rows[0].columns if col.startswith('pn_')]
        self.acol = None
        self.pncols = OrderedDict()
        for r in self.rows:
            for pncol in pncols:
                del r[pncol]

    def pn(self, col, n_or_fn):
        "portfolio numbering for independent sort"
        pncol = 'pn_' + col
        for r in self:
            r[pncol] = ''

        # parentheses around lambda are necessary
        fn = (lambda rs: nchunks(rs, n_or_fn)) \
             if isinstance(n_or_fn, int) else n_or_fn
        for rs1 in self.num(col).order(self.dcol).group(self.dcol):
            for pn, rs2 in enumerate(fn(rs1.order(col)), 1):
                for r in rs2:
                    r[pncol] = pn

        self.pncols[pncol] = pn
        return self

    def dpn(self, col, n_or_fn, pncols=None):
        """
        portfolio numbering for dependent sort
        if you don't specify pncols, self.pncols is used
        """
        pncol = 'pn_' + col

        for r in self:
            r[pncol] = ''

        # parentheses around lambda are necessary
        fn = (lambda rs: nchunks(rs, n_or_fn)) \
             if isinstance(n_or_fn, int) else n_or_fn

        pncols = listify(pncols) if pncols else list(self.pncols)

        for rs1 in self.num(pncols + [col]).order(self.dcol).group(self.dcol):
            for rs2 in rs1.order(pncols + [col]).group(pncols):
                for pn, rs3 in enumerate(fn(rs2), 1):
                    for r in rs3:
                        r[pncol] = pn

        self.pncols[pncol] = pn
        return self

    def pn1(self, col, n_or_fn):
        """
        Assign portfolios based on the first date numbering
        Ex) As in making factors
        """
        pncol = 'pn_' + col
        for r in self:
            r[pncol] = ''
        # first date rows
        fdrows = next(self.num(col).order(self.dcol).group(self.dcol))
        # assign it first
        PRows(fdrows, self.dcol, self.fcol).pn(col, n_or_fn)
        self._assign_follow_ups(col)
        self.pncols[pncol] = n_or_fn if isinstance(n_or_fn, int) else \
                             len(n_or_fn(fdrows))
        return self

    def dpn1(self, col, n_or_fn, pncols=None):
        """
        dependent version of pn1
        """
        # most of the code is just a dups with pn1 but they are simple enough
        # not to cut out in other places.
        pncol = 'pn_' + col
        for r in self:
            r[pncol] = ''
        # first date rows
        fdrows = next(self.num(col).order(self.dcol).group(self.dcol))
        # assign it first
        PRows(fdrows, self.dcol, self.fcol).dpn(col, n_or_fn, pncols)
        self._assign_follow_ups(col)
        self.pncols[pncol] = n_or_fn if isinstance(n_or_fn, int) else \
                             len(n_or_fn(fdrows))

        return self

    def pns(self, *colns):
        return self._pns_helper(self.pn, colns)

    def dpns(self, *colns):
        return self._pns_helper(self.dpn, colns)

    def pns1(self, *colns):
        return self._pns_helper(self.pn1, colns)

    def dpns1(self, *colns):
        return self._pns_helper(self.dpn1, colns)

    def _pns_helper(self, pnfn, colns):
        self.reset()
        colns1 = list(grouper(colns, 2))
        pnfn(*colns1[0])
        for col, n in colns1[1:]:
            pnfn(col, n)
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
            result.append(Rows(rs))

        line = []
        line.append(self.acol)
        for rs in result:
            line.append(_mrep(rs, self.acol))
        seq = [h - l for h, l in \
               zip(result[-1][self.acol], result[0][self.acol])]
        line.append(_mrep0(seq))
        return Box([line])

    def _pat2(self, pncol1, pncol2):
        result = []
        for rs1 in self.order([pncol1, pncol2, self.dcol]).group(pncol1):
            result1 = []
            for rs2 in rs1.group(pncol2):
                result1.append(Rows(rs2))
            result.append(result1)

        line = []
        line.append("%s\\%s" % (pncol1[3:], pncol2[3:]))
        for i in range(1, len(result[0]) + 1):
            line.append(i)
        line.append('P%s - P1' % i)

        lines = []
        lines.append(line)

        for i, result1 in enumerate(result, 1):
            line = []
            line.append(i)
            for rs in result1:
                line.append(_mrep(rs, self.acol))
            seq = [h - l for h, l in \
                   zip(result1[-1][self.acol], result1[0][self.acol])]
            line.append(_mrep0(seq))
            lines.append(line)

        # bottom line
        line = []
        line.append("P%s - P1" % (i,))
        seqs = []
        for hseq, lseq in zip(result[-1], result[0]):
            seq = [h - l for h, l in zip(hseq[self.acol], lseq[self.acol])]
            seqs.append(seq)
            line.append(_mrep0(seq))
        # bottom right corner
        line.append(_mrep0([h - l for h, l in zip(seqs[-1], seqs[0])]))

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
        lines.append([_mrep0(self[col]) for col in cols])
        return Box(lines)

    def famac(self, model):
        "Fama Macbeth"
        def allvars(model):
            left, right = model.split('~')
            return [left.strip()] + [x.strip() for x in right.split('+')]

        xvs = ['intercept'] + allvars(model)[1:]

        params = []
        for rs1 in self.order(self.dcol).group(self.dcol):
            rs1 = rs1.num(allvars(model))
            if len(rs1) >= 2:
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

    def roll(self, period, jump, begdate=None, enddate=None):
        "group rows over time, allowing overlaps"
        def get_nextdate(date, period):
            "date after the period"
            date = str(date)
            if len(date) == 8:
                return yyyymmdd(date, period)
            elif len(date) == 6:
                return yyyymm(date, period)
            elif len(date) == 4:
                return int(date) + period
            else:
                raise ValueError('Invalid date', date)

        begdate = int(begdate) if begdate else self.rows[0][self.dcol]
        enddate = int(enddate) if enddate else self.rows[-1][self.dcol]

        while begdate <= enddate:
            yield self.between(begdate, get_nextdate(begdate, period))
            begdate = get_nextdate(begdate, jump)

    def between(self, beg, end):
        "begdate <= x <  enddate"
        return self.where(lambda r: r[self.dcol] >= beg and r[self.dcol] < end)

    def _assign_follow_ups(self, col):
        """assign portfolio numbers based on the first date values

        col: column to assign portfolio number
        fcol: firm id column name
        """
        pncol = 'pn_' + col
        for rs1 in self.num(col).order([self.fcol, self.dcol]).group(self.fcol):
            pn = rs1[0][pncol]
            if isinstance(pn, int):
                for r in rs1[1:]:
                    r[pncol] = pn


def _mrep(rs, col):
    "mean representation"
    m = round(st.mean(r[col] for r in rs), 3)
    n = round(st.mean(r.n for r in rs))
    return "%s (%s)" % (m, n)


def _mrep0(seq):
    "sequence of numbers with t val"
    tstat = ttest_1samp(seq, 0)
    return "%s [%s]" % (star(st.mean(seq), tstat[1]), round(tstat[0], 2))


def wavg(rs, col, wcol):
    "compute weigthed average"
    rs = [r for r in rs if isnum(r[col]) and isnum(r[wcol])]
    total = sum(r[wcol] for r in rs)
    return sum(r[col] * r[wcol] / total for r in rs)


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
