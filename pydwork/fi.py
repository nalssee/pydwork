"""
Frequently used financial data work
"""

import statistics as st

from itertools import combinations, takewhile, dropwhile, product

from scipy.stats import ttest_1samp

from pydwork.sqlplus import Rows, Row
from .util import nchunks, listify, yyyymm, yyyymmdd


class PRows(Rows):
    """Rows for portfolio analysis
    """
    def __init__(self, rows, dcol):
        Rows.__init__(self, rows)
        # date column
        self._dcol = dcol
        # (computed) average column
        self._acol = None
        # portfolio sizes
        self._pncols = []

    def pn(self, col, n):
        "portfolio numbering for independent sort"
        pncol = 'pn_' + col
        for r in self:
            r[pncol] = ''

        self.num(col)
        for rs1 in self.order(self._dcol).group(self._dcol):
            for pn, rs2 in enumerate(nchunks(rs1.order(col), n), 1):
                for r in rs2:
                    r[pncol] = pn

        if (pncol, n) not in self._pncols:
            self._pncols.append((pncol, n))
        return self

    def dpn(self, col, n):
        """
        portfolio numbering for dependent sort
        """
        pncol = 'pn_' + col

        for r in self:
            r[pncol] = ''

        pncols = [pncol for pncol, _ in self._pncols]

        self.num(pncols + [col])
        for rs1 in self.order(self._dcol).group(self._dcol):
            for rs2 in rs1.order(pncols + [col]).group(pncols):
                for pn, rs3 in enumerate(nchunks(rs2, n), 1):
                    for r in rs3:
                        r[pncol] = pn

        if (pncol, n) not in self._pncols:
            self._pncols.append((pncol, n))
        return self

    def avg(self, col, wcol=None, pncols=None):
        "wcol: weight column"
        self.num([col] + [wcol]) if wcol else self.num(col)

        if pncols:
            pncols = listify(pncols)
        else:
            pncols = [pncol for pncol, _ in self._pncols]

        ns = [range(1, n + 1) for pncol, n in self._pncols if pncol in pncols]

        result = []
        for rs1 in self.group(self._dcol):
            for pns, rs2 in zip(product(*ns), rs1.order(pncols).group(pncols)):
                # test if there's any missing portfolio
                if [rs2[0][pncol] for pncol in pncols] != list(pns):
                    raise ValueError('missing portfolio no. %s in %s' %  \
                        (list(pns), rs2[0][self._dcol]))

                r = Row()
                r[self._dcol] = rs2[0][self._dcol]
                for pncol, pn in zip(pncols, pns):
                    r[pncol] = pn
                r.n = len(rs2)

                if wcol:
                    r[col] = wavg(rs2, col, wcol)
                else:
                    r[col] = st.mean(rs2[col])

                result.append(r)

        prows = PRows(result, self._dcol)
        prows._pncols = pncols
        prows._acol = col
        return prows

    def pshow(self, pncols=None):
        "show pattern"
        if pncols:
            pncols = listify(pncols)
        else:
            pncols = [pncol for pncol, _ in self._pncols]

        if len(pncols) == 1:
            self._phsow()


        result = []
        for rs1 in self.order(pncols).group(pncols):
            r = Row()
            seq = [r.avg for r in rs1]
            r.pn = rs1[0].pn
            r.avg = st.mean(seq)
            tstat = ttest_1samp(seq, 0)
            r.tval = tstat[0]
            r.pval = tstat[1]

            result.append(r)


def wavg(rs, col, wcol):
    "compute weigthed average"
    total = sum(r[wcol] for r in rs)
    return st.mean(r[col] * r[wcol] / total for r in rs)


# continues from avg_pn
def avg_pts(rs, datecol):
    "avg over time series of portfolio values"
    def fill(ns, ns_all):
        x = (set(ns_all) - set(ns)).pop()
        return x, ns + (x,)

    def keyfn(ns, datecol=None):
        def fn(r):
            pn = _decode_pns(r.pn)
            if datecol:
                return [pn[n] for n in ns] + [r[datecol]]
            else:
                return [pn[n] for n in ns]
        return fn

    result = []
    for rs1 in rs.order('pn').group('pn'):
        r = Row()
        seq = [r.avg for r in rs1]
        r.pn = rs1[0].pn
        r.avg = st.mean(seq)
        tstat = ttest_1samp(seq, 0)
        r.tval = tstat[0]
        r.pval = tstat[1]

        result.append(r)

    npn = len(_decode_pns(rs[0].pn))

    for ns1 in combinations(range(1, npn), npn - 2):
        missing, ns2 = fill(ns1, range(1, npn))

        for rs1 in rs.order(keyfn(ns2, datecol)).group(keyfn(ns1)):

            grs = list(rs1.group(lambda r: _decode_pns(r.pn)[missing]))

            high = grs[-1]
            low = grs[0]
            seq = [r1.avg - r2.avg for r1, r2 in zip(high, low)]
            r = Row()
            r.pn = high[0].pn + ' - ' + low[0].pn
            r.avg = st.mean(seq)
            tstat = ttest_1samp(seq, 0)
            r.tval = tstat[0]
            r.pval = tstat[1]

            result.append(r)

    return Rows(result)


def famac(rs, model, datecol, show=True):
    "Fama Macbeth"
    def xvars(model):
        _, right = model.split('~')
        return [x.strip() for x in right.split('+')]

    xvs = ['intercept'] + xvars(model)

    params = []
    for rs1 in rs.order(datecol).group(datecol):
        reg = rs1.ols(model)
        r = Row()
        r[datecol] = rs1[0][datecol]
        for var, p in zip(xvs, reg.params):
            r[var] = p
        r.n = int(reg.nobs)
        r.r2 = reg.rsquared
        params.append(r)

    params1 = Rows(params)

    if show:
        print('var,avg,tval')
        for var in xvs:
            seq = params1[var]
            tval = ttest_1samp(seq, 0)
            print(var, end=',')
            print(star(st.mean(seq), tval[1]), end=',')
            print(round(tval[0], 2))

    return params1


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


def _dim1print(rs):
    "1 dim print"
    print('', end=',')
    for i in range(1, len(rs)):
        print(i, end=',')
    print('high - low (tval)')
    print('avg', end=',')
    for r in rs[:-1]:
        print(star(r.avg, r.pval), end=',')
    r0 = rs[-1]
    print('%s (%s)' % (star(r0.avg, r0.pval), round(r0.tval, 2)))


def _dim2print(rs):
    "2 dim print"
    rs1, rs2 = _nonhilo(rs), _hilo(rs)
    grs = list(_nonhilo(rs1).group(lambda r: r.pn[:3]))
    n, k = len(grs), len(grs[0])
    print('', end=',')
    for i in range(1, k + 1):
        print(i, end=',')
    print('high - low (tval)')
    for i, (rs0, hl) in enumerate(zip(grs, rs2), 1):
        print(i, end=',')
        for r in rs0:
            print(star(r.avg, r.pval), end=',')
        print('%s (%s)' % (star(hl.avg, hl.pval), round(hl.tval, 2)))
    print('high - low', end=',')
    for r in rs2[n:]:
        print('%s (%s)' % (star(r.avg, r.pval), round(r.tval, 2)), end=',')
    print()


def _hilo(rs):
    "(high - low) rows"
    return rs.where(lambda r: '-' in r.pn)


def _nonhilo(rs):
    "non (high - low) rows, ordinary rows"
    return rs.where(lambda r: '-' not in r.pn)


def pprint(rs):
    "pattern print, 1d and 2d"
    dim = len(rs[0].pn.split('/')) - 1
    if dim == 1:
        _dim1print(rs)
    elif dim == 2:
        _dim2print(rs)
    else:
        raise ValueError('Dimension must be 1 or 2')


def overlap(rs, datecol, period, jump):
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
        return Rows(takewhile(lambda r: r[datecol] < enddate, rs))

    while rs != []:
        startdate = str(rs[0][datecol])
        enddate = get_nextdate(startdate, jump)
        yield rows_for(rs, startdate, period)
        rs = Rows(dropwhile(lambda r: r[datecol] < enddate, rs))
