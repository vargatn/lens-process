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


def xpatches(raw_patches):
    infos = []
    datas = []
    labels = []
    for i, patch in enumerate(raw_patches):
        if len(patch) > 0:
            if len(patch.shape) == 1:
                info, data, tmp = xread(patch[np.newaxis, :])
                # info, data, tmp = xsample(patch[np.newaxis, :])
            else:
                info, data, tmp = xread(patch)
                # info, data, tmp = xsample(patch)
            infos.append(info)
            datas.append(data)
            labels.append(np.ones(info.shape[0]) * i)
    infos = np.vstack(infos)
    datas = np.concatenate(datas, axis=1)
    labels = np.concatenate(labels)
    return infos, datas, labels


def xread(xdata, **kwargs):
    """Loader for xshear output"""
    valnames = {
        "info": ("index", "weight_tot", "totpairs"),
        "data": ("npair_i", "rsum_i", "wsum_i", "ssum_i", "dsum_i", "osum_i", "dsensum_w_i", "osensum_w_i",
                 "dsensum_s_i", "osensum_s_i"),
    }

    # calculates number of radial bins used
    # print xdata.shape
    bins = (xdata.shape[1] - 3) // 10
    # print bins
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
    # print info[:, 2], np.sum(data[0, :, :], axis=1)
    assert (info[:, 2] == np.sum(data[0, :, :], axis=1)).all()

    return info, data, valnames





