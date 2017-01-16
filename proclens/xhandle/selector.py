"""

"""

import numpy as np


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


def safedivide(x, y, eps=1e-8):
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
