# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
Statistic functions.
"""

import numpy as np


def vcorrcoef(X, y, dim):
    """
    Compute vectorized correlation coefficient.
    refer to:
    https://waterprogramming.wordpress.com/2014/06/13/numpy-vectorized-correlation-coefficient/

    :param X: nD array.
    :param y: 1D array.
    :param dim: along dimension to compute correlation coefficient.
    :return: correlation coefficient array.
    """

    X = np.array(X)
    ndim = X.ndim

    # roll lat dim axis to last
    X = np.rollaxis(X, dim, ndim)

    Xm = np.mean(X, axis=-1, keepdims=True)
    ym = np.mean(y)
    r_num = np.sum((X - Xm) * (y - ym), axis=-1)
    r_den = np.sqrt(np.sum((X - Xm) ** 2, axis=-1) * np.sum((y - ym) ** 2))
    r = r_num / r_den

    return r
