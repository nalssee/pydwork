"""
Frequently used financial data work
"""

import statistics as st

from itertools import product
from collections import OrderedDict

from scipy.stats import ttest_1samp

from .sqlplus import Rows, Row, Box
from .util import nchunks, listify, grouper, yyyymm, yyyymmdd,\
    parse_model, star


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

    def pn(self, *colns):
        """indenpendent sort,
        ex) self.pn('a', 10, 'b', lambda rs: [rs[:4], rs[4:]])"""
        for col, n in grouper(colns, 2):
            self._pn(col, n)
        return self

    def _pn(self, col, nfn):
        pncol = 'pn_' + col
        self[pncol] = ''

        if isinstance(nfn, int):
            nfn = (lambda nfn: lambda rs: nchunks(rs, nfn))(nfn)
        for rs1 in self.num(col).order(self.dcol).group(self.dcol):
            for pn, rs2 in enumerate(nfn(rs1.order(col)), 1):
                for r in rs2:
                    r[pncol] = pn
        if pn:
            self.pncols[pncol] = pn
        else:
            raise ValueError("No portfolios formed")
        return self

    def dpn(self, *colns):
        "dependent sort"
        for col, n in grouper(colns, 2):
            self._dpn(col, n)
        return self

    def _dpn(self, col, nfn):
        pncol = 'pn_' + col
        self[pncol] = ''

        if isinstance(nfn, int):
            nfn = (lambda nfn: lambda rs: nchunks(rs, nfn))(nfn)
        pncols = list(self.pncols)
        for rs1 in self.num(pncols + [col]).order(self.dcol).group(self.dcol):
            for rs2 in rs1.order(pncols + [col]).group(pncols):
                for pn, rs3 in enumerate(nfn(rs2), 1):
                    for r in rs3:
                        r[pncol] = pn
        if pn:
            self.pncols[pncol] = pn
        else:
            raise ValueError("No portfolios formed")
        return self

    # Number portfolios as you roll
    # Just this time portfolio numbers are based on the first date values
    # all the others simply follow the first one
    # This will be useful when you make factor portfolios
    def pnroll(self, period, *colns):
        return self._pnroll('pn', period, colns)

    def dpnroll(self, period, *colns):
        return self._pnroll('dpn', period, colns)

    def _pnroll(self, pnfn, period, colns):
        "pnfn: method name string, dpn or pn"
        assert self.fcol is not None, "fcol required"

        cols = [col for col, _ in grouper(colns, 2)]
        pncols = ['pn_' + col for col in cols]
        self[pncols] = ''
        for rs in self.order(self.dcol).roll(period, period):
            # first date rows
            fdrows = PRows(next(rs.group(self.dcol)), self.dcol, self.fcol)
            getattr(fdrows, pnfn)(*colns)
            for rs1 in rs.order([self.fcol, self.dcol]).group(self.fcol):
                rs1[pncols] = [rs1[0][pncol] for pncol in pncols]
        self.pncols = fdrows.pncols
        return self

    def pavg(self, col, wcol=None, pncols=None):
        "portfolio average,  wcol: weight column"
        self.is_valid()
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
                    raise ValueError('missing portfolio no. %s %s in a%s a ' %
                                     (list(pns),[rs2[0][pncol] for pncol in pncols], rs2[0][self.dcol]))

                r = Row()
                r[self.dcol] = rs2[0][self.dcol]
                for pncol, pn in zip(pncols, pns):
                    r[pncol] = pn
                r.n = len(rs2)
                r[col] = rs2.wavg(col, wcol)
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
        elif len(pncols) > 2:
            return self._patn(pncols)
        else:
            raise ValueError("Invalid pncols")

    def _pat1(self, pncol):
        result = []
        for rs in self.order([pncol, self.dcol]).group(pncol):
            result.append(Rows(rs))

        line = []
        line.append(self.acol)
        for rs in result:
            line.append(_mrep(rs, self.acol))
        seq = [h - l for h, l in
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
            seq = [h - l for h, l in
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
        xvs = ['intercept'] + parse_model(model)[1:]
        params = []
        for rs1 in self.order(self.dcol).group(self.dcol):
            rs1 = rs1.num(parse_model(model))
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

    def between(self, beg, end=None):
        "begdate <= x <  enddate"
        if end:
            return self.where(lambda r: r[self.dcol] >= beg and
                              r[self.dcol] < end)
        else:
            return self.where(lambda r: r[self.dcol] >= beg)


def _mrep(rs, col):
    "mean representation"
    m = round(st.mean(r[col] for r in rs), 3)
    n = round(st.mean(r.n for r in rs))
    return "%s (%s)" % (m, n)


def _mrep0(seq):
    "sequence of numbers with t val"
    tstat = ttest_1samp(seq, 0)
    return "%s [%s]" % (star(st.mean(seq), tstat[1]), round(tstat[0], 2))
