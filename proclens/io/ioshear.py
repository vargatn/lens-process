"""

"""

import numpy as np


def makecat(fname, mid, ra, dec, z):
    """
    Write an xshear style lens catalog to file

    :param fname: path to file
    :param mid: ID of object (int)
    :param ra: RA (float)
    :param dec: DEC (float)
    :param z: redshift (float)
    """

    table = np.vstack((mid, ra, dec, z, np.ones(shape=z.shape))).T
    fmt = ['%d',] + 3 * ['%.6f',] + ['%d',]
    np.savetxt(fname, table, fmt = fmt)


def xhandler(xdata, xmode='reduced', **kwargs):
    """
    Converts xshear output table into a readable format

    columns:
    ---------
    index, weight_tot, totpairs, npair_i, rsum_i, wsum_i, dsum_i, osum_i,
    dsensum_i, osensum_i

    (for lensfit mode, in reduced mode dsensum_i and osensum_i is replaced
     with wsum_i)

    description:
    -------------
    index:      index from lens catalog
    weight_tot: sum of all weights for all source pairs in all radial bins
    totpairs:   total pairs used
    npair_i:    number of pairs in radial bin i.  N columns.
    rsum_i:     sum of radius in radial bin i
    wsum_i:     sum of weights in radial bin i
    dsum_i:     sum of \Delta\Sigma_+ * weights in radial bin i.
    osum_i:     sum of \Delta\Sigma_x * weights in  radial bin i.
    dsensum_i:  sum of weights times sensitivities
    osensum_i:  sum of weights times sensitivities

    :param xdata: xshear data file

    :param xmode: switches different output file formats:
                  point, reduced, sample

    :returns: info, data, valnames
    """

    if xmode == 'reduced':
        info, data, valnames = xreduced(xdata)
    elif xmode == 'sample':
        info, data, valnames = xsample(xdata)
    elif xmode == 'lensfit':
        info, data, valnames = xlensfit(xdata)
    else:
        raise ValueError('invalid type specified')

    return info, data, valnames


def xsample(xdata, **kwargs):
    """Loader for reduced-style xshear output"""
    valnames = {
        "info": ("index", "weight_tot", "totpairs"),
        "data": ("npair_i", "rsum_i", "wsum_i", "dsum_i", "osum_i",
                 'wscinvsum_i', 'wscinvsum_i'),
    }

    # calculates number of radial bins used
    bins = (xdata.shape[1] - 3) // 6
    print bins
    # position indexes
    sid = 3
    pos_npair = 0
    pos_rsum = 1
    pos_wsum = 2
    pos_dsum = 3
    pos_osum = 4
    pos_scinv = 5

    gid = xdata[:, 0]
    weight_tot = xdata[:, 1]
    tot_pairs = xdata[:, 2]
    npair = xdata[:, sid + pos_npair * bins: sid + (pos_npair + 1) * bins]
    rsum = xdata[:, sid + pos_rsum * bins: sid + (pos_rsum + 1) * bins]
    wsum = xdata[:, sid + pos_wsum * bins: sid + (pos_wsum + 1) * bins]
    scinv = xdata[:, sid + pos_scinv * bins: sid + (pos_scinv + 1) * bins]
    dsum = xdata[:, sid + pos_dsum * bins: sid + (pos_dsum + 1) * bins]
    osum = xdata[:, sid + pos_osum * bins: sid + (pos_osum + 1) * bins]

    info = np.vstack((gid, weight_tot, tot_pairs)).T

    data = np.dstack((npair, rsum, wsum, dsum, osum, scinv, scinv))
    data = np.transpose(data, axes=(2, 0, 1))

    # checking if loading made sense
    print info[:, 2], np.sum(data[0, :, :], axis=1)
    # assert (info[:, 2] == np.sum(data[0, :, :], axis=1)).all()

    # return info, data, valnames


def xread(xdata, **kwargs):
    """Loader for xshear output"""
    valnames = {
        "info": ("index", "weight_tot", "totpairs"),
        "data": ("npair_i", "rsum_i", "wsum_i", "ssum_i", "dsum_i", "osum_i", "dsensum_w_i", "osensum_w_i",
                 "dsensum_s_i", "osensum_s_i"),
    }

    # calculates number of radial bins used
    bins = (xdata.shape[1] - 3) // 10
    print bins

    # position indexes
    sid = 3
    pos_npair = 0
    pos_rsum = 1
    pos_wsum = 2
    pos_ssum = 3
    pos_dsum = 4
    pos_osum = 5
    pos_dsensum_w = 6
    pos_osensum_w = 7
    pos_dsensum_s = 8
    pos_osensum_s = 9

    gid = xdata[:, 0]
    weight_tot = xdata[:, 1]
    tot_pairs = xdata[:, 2]

    npair = xdata[:, sid + pos_npair * bins: sid + (pos_npair + 1) * bins]
    rsum = xdata[:, sid + pos_rsum * bins: sid + (pos_rsum + 1) * bins]
    wsum = xdata[:, sid + pos_wsum * bins: sid + (pos_wsum + 1) * bins]
    ssum = xdata[:, sid + pos_ssum * bins: sid + (pos_ssum + 1) * bins]
    dsum = xdata[:, sid + pos_dsum * bins: sid + (pos_dsum + 1) * bins]
    osum = xdata[:, sid + pos_osum * bins: sid + (pos_osum + 1) * bins]
    dsensum_w = xdata[:, sid + pos_dsensum_w * bins: sid + (pos_dsensum_w + 1) * bins]
    osensum_w = xdata[:, sid + pos_osensum_w * bins: sid + (pos_osensum_w + 1) * bins]
    dsensum_s = xdata[:, sid + pos_dsensum_s * bins: sid + (pos_dsensum_s + 1) * bins]
    osensum_s = xdata[:, sid + pos_osensum_s * bins: sid + (pos_osensum_s + 1) * bins]


    info = np.vstack((gid, weight_tot, tot_pairs)).T
    data = np.dstack((npair, rsum, wsum, ssum, dsum, osum, dsensum_w, osensum_w, dsensum_s, osensum_s))
    data = np.transpose(data, axes=(2, 0, 1))

    # checking if loading made sense
    assert (info[:, 2] == np.sum(data[0, :, :], axis=1)).all()

    return info, data, valnames


def xreduced(xdata, **kwargs):
    """Loader for reduced-style xshear output"""
    valnames = {
        "info": ("index", "weight_tot", "totpairs"),
        "data": ("npair_i", "rsum_i", "wsum_i", "dsum_i", "osum_i",
                 "wsum_i", "wsum_i"),
    }

    # calculates number of radial bins used
    bins = (xdata.shape[1] - 3) // 5

    # position indexes
    sid = 3
    pos_npair = 0
    pos_rsum = 1
    pos_wsum = 2
    pos_dsum = 3
    pos_osum = 4

    gid = xdata[:, 0]
    weight_tot = xdata[:, 1]
    tot_pairs = xdata[:, 2]
    npair = xdata[:, sid + pos_npair * bins: sid + (pos_npair + 1) * bins]
    rsum = xdata[:, sid + pos_rsum * bins: sid + (pos_rsum + 1) * bins]
    wsum = xdata[:, sid + pos_wsum * bins: sid + (pos_wsum + 1) * bins]
    dsum = xdata[:, sid + pos_dsum * bins: sid + (pos_dsum + 1) * bins]
    osum = xdata[:, sid + pos_osum * bins: sid + (pos_osum + 1) * bins]

    info = np.vstack((gid, weight_tot, tot_pairs)).T
    data = np.dstack((npair, rsum, wsum, dsum, osum, wsum, wsum))
    data = np.transpose(data, axes=(2, 0, 1))

    # checking if loading made sense
    assert (info[:, 2] == np.sum(data[0, :, :], axis=1)).all()

    return info, data, valnames


def xlensfit(xdata, **kwargs):
    """Loader for lensfit-style xshear output"""
    valnames = {
        "info": ("index", "weight_tot", "totpairs"),
        "data": ("npair_i", "rsum_i", "wsum_i", "dsum_i", "osum_i",
                 "dsensum_i", "osensum_i"),
    }

    # calculates number of radial bins used
    bins = (xdata.shape[1] - 3) // 7

    # position indexes
    sid = 3
    pos_npair = 0
    pos_rsum = 1
    pos_wsum = 2
    pos_dsum = 3
    pos_osum = 4
    pos_dsensum = 5
    pos_osensum = 6

    gid = xdata[:, 0]
    weight_tot = xdata[:, 1]
    tot_pairs = xdata[:, 2]
    npair = xdata[:, sid + pos_npair * bins: sid + (pos_npair + 1) * bins]
    rsum = xdata[:, sid + pos_rsum * bins: sid + (pos_rsum + 1) * bins]
    wsum = xdata[:, sid + pos_wsum * bins: sid + (pos_wsum + 1) * bins]
    dsum = xdata[:, sid + pos_dsum * bins: sid + (pos_dsum + 1) * bins]
    osum = xdata[:, sid + pos_osum * bins: sid + (pos_osum + 1) * bins]
    dsensum = xdata[:,
              sid + pos_dsensum * bins: sid + (pos_dsensum + 1) * bins]
    osensum = xdata[:,
              sid + pos_osensum * bins: sid + (pos_osensum + 1) * bins]

    info = np.vstack((gid, weight_tot, tot_pairs)).T
    data = np.dstack((npair, rsum, wsum, dsum, osum, dsensum, osensum))
    data = np.transpose(data, axes=(2, 0, 1))

    # checking if loading made sense
    assert (info[:, 2] == np.sum(data[0, :, :], axis=1)).all()

    return info, data, valnames



