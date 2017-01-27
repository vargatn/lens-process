
import numpy as np

def redges(rmin, rmax, nbin):
    """
    Calculates nominal edges and centers for logarithmic bins
    (base10 logarithm is used)

    Edges and areas are exact, "center" values are estimated assuming a
    uniform source surface density. That is it gives the CIRCUMFERENCE weighted
    radius...

    :param rmin: inner edge

    :param rmax: outer edge

    :param nbin: number of bins

    :returns: centers, edges, areas

    """
    edges = np.logspace(math.log10(rmin), math.log10(rmax), nbin + 1,
                        endpoint=True)
    cens = np.array([(edges[i + 1] ** 3. - edges[i] ** 3.) * 2. / 3. /
                     (edges[i + 1] ** 2. - edges[i] ** 2.)
                     for i, edge in enumerate(edges[:-1])])

    areas = np.array([np.pi * (edges[i + 1] ** 2. - edges[i] ** 2.)
                      for i, val in enumerate(edges[:-1])])

    return cens, edges, areas