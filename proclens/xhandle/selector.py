"""

"""

import numpy as np
import kmeans_radec as krd
import itertools

def index3d(i, j, k, jmax, kmax):
    return k  + (kmax) * j + (jmax) * kmax * i

def partition(lst, n):
    """Divides the list into n chunks"""
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))]
            for i in range(n) ]


def col_attach(table, colname, dtype, col):
    """
    A brute forece sideways stack for record arrays


    Mostly a convinience method

    :param table: Table to append sideways
    :param colname: name of the column
    :param dtype: dtype of the column
    :param col: data to be added to the column
    :return: updated table
    """

    coltypes = table.dtype.descr + [(colname, dtype)]
    res = np.zeros(len(table), dtype=coltypes)
    for ct in coltypes[:-1]:
        res[ct[0]] = table[ct[0]]
    res[colname] = col

    return res


def get_jklabel(ra, dec, centers):
    """
    Assigns a jk label to the points based on the passed centers

    :param ra: RA
    :param dec: DEC
    :param centers: coordinates of centers for K-means patches
    :return: (inds which are NOT IN patch i, inds whihc are IN patch i
    """

    pos = np.vstack((ra, dec)).T

    km = krd.KMeans(centers)

    labels = km.find_nearest(pos).astype(int)


    sub_labels = np.arange(len(centers), dtype=int)
    # sub_labels = np.unique(labels)

    # indexes of clusters for subsample i
    non_indexes = [np.where(labels != ind)[0] for ind in sub_labels]

    # indexes of clusters not in subsample i
    indexes = [np.where(labels == ind)[0] for ind in sub_labels]

    return indexes, non_indexes, labels


def selector(pps, limits):
    """
    Applies selection to array based on the passed parameter limits

    :param pp: numpy table of values array
    :param limits1: list of list of corresponding limits, on list of limits for each column of pp
    :returns: list of indices, (tuple of inputs)
    """

    parinds = np.arange(pps.shape[1], dtype=int)
    plpairs = []
    for pp, limit in zip(parinds, limits):
        plpair = []
        for i, lim in enumerate(limit[:-1]):
            plpair.append((pp, (limit[i], limit[i + 1])))
        plpairs.append(plpair)

    sinds = []
    for pval in itertools.product(*plpairs):
        sval = np.zeros(len(pps), dtype=bool)
        for (pind, limit) in pval:
            sval += (limit[0] <= pps[:, pind]) * (pps[:, pind] < limit[1])
        sinds.append(sval)

    sinds = np.array(sinds)
    return sinds


def getselect(pp1, pp2, limits1, limits2):
    """
    Applies selection to array based on the passed parameter limits


    :param pp1: param1
    :param pp2: param2
    :param limits1: edges for param1
    :param limits2: edges for param2
    :returns: list of indices, (tuple of inputs)
    """
    sinds = []
    for j, zlim in enumerate(limits2[:-1]):
        for i, llim in enumerate(limits1[:-1]):
            inds = (limits2[j] < pp2) * (pp2 <= limits2[j + 1]) * (limits1[i] < pp1) * (pp1 <= limits1[i + 1])
            sinds.append(inds)
    sinds = np.array(sinds)

    vals1 = np.arange(len(limits1))
    vals2 = np.arange(len(limits2))
    grid1, grid2 = np.meshgrid(vals1, vals2)
    arr1 = grid1.flatten()
    arr2 = grid2.flatten()
    return sinds, (arr1, arr2, limits1, limits2)


def safedivide(x, y, eps=1e-12):
    """calculates x/ y for arrays, with an attempt to handle zeros in the denominater by setting the result to zero"""
    xabs = np.abs(x)
    yabs = np.abs(y)
    gind = np.where((xabs > eps) * (yabs > eps))

    res = np.zeros(shape=xabs.shape)
    res[gind] = x[gind] / y[gind]
    return res


def digitizedd(x, bins):
    """D-dimenisonal digitize"""
    if len(bins) == 1:
        return np.digitize(x, bins)
    pos = []
    for i, edges in enumerate(bins):
        pos.append(np.digitize(x[:, i], bins[i]))
    return np.array(pos).T


def match2d(pars, refpars, win=None, wref=None, nbin=30):
    """Matches dist2 to dist1"""
    refhist = np.histogram2d(refpars[:, 0], refpars[:, 1], bins=nbin, weights=wref)
    refcounts = refhist[0]
    bin_edges = (refhist[1], refhist[2])

    parcounts = np.histogram2d(pars[:, 0], pars[:, 1], bins=bin_edges, weights=win)[0]
    # relative weight for matching
    wratio = safedivide(refcounts, parcounts)

    digits = np.vstack((np.digitize(pars[:, 0], bins=bin_edges[0]),
                        np.digitize(pars[:, 1], bins=bin_edges[1]))).T

    # assigning weights to individual random points
    ww = np.zeros(len(pars))
    for i, dig in enumerate(digits):
        if dig[0] != 0 and dig[1] != 0 and dig[0] != len(bin_edges[0]) and dig[1] != len(bin_edges[1]):
            ww[i] = wratio[dig[0] - 1, dig[1] - 1]
            if win is not None:
                ww[i] *= win[i]
    return ww