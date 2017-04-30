"""
Handles Fits -> Pandas transformations

FITS tables sometimes have multidimensional columns, which are not supported for DataFrames

Pandas DataFrames however provide many nice features, such as SQL speed database matchings
"""



import numpy as np
import pandas as pd
import esutil.numpy_util as nu
import copy
from astropy.table import Table


def flat_type(recarr):
    mags = ['G', 'R', 'I', 'Z']

    newtype = []
    for dt in recarr.dtype.descr:
        if len(dt) == 3:
            for i in np.arange(dt[2][0]):
                if 'MAG' in dt[0]:
                    newtype.append((dt[0] + '_' + mags[i], dt[1]))
                else:
                    newtype.append((dt[0] + '_' + str(i), dt[1]))
        else:
            newtype.append(dt)
    return newtype


def flat_copy(recarr):
    newtype = flat_type(recarr)
    newarr = np.zeros(len(recarr), dtype=newtype)

    oldnames = recarr.dtype.names
    j = 0
    for i, dt in enumerate(recarr.dtype.descr):
        if len(dt) == 3:
            for c in np.arange(dt[2][0]):
                #                 print newtype[j]
                newarr[newtype[j][0]] = recarr[oldnames[i]][:, c]
                j += 1

        else:
            #             print newtype[j]
            newarr[newtype[j][0]] = recarr[oldnames[i]]
            j += 1
    return newarr


def to_pandas(recarr):
    newarr = flat_copy(recarr)
    res = pd.DataFrame.from_records(newarr.byteswap().newbyteorder(), columns=newarr.dtype.names)
    return res

