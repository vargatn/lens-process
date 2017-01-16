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