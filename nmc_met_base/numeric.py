# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
Numeric number manipulation.
"""

import math
import numpy as np


def ensure_numeric(A, typecode=None):
    """
    Ensure that sequence is a numeric array.

    :param A: Sequence. If A is already a numeric array it will be returned
                        unaltered.
                        If not, an attempt is made to convert it to a
                        numeric array.
           A: Scalar.   Return 0-dimensional array containing that value. Note
                        that a 0-dim array DOES NOT HAVE A LENGTH UNDER numpy.
           A: String.   Array of ASCII values (numpy can't handle this)
    :param typecode: numeric type. If specified, use this in the conversion.
                     If not, let numeric package decide.
                     typecode will always be one of num.float, num.int, etc.
    :return:
    """

    if isinstance(A, str):
        msg = 'Sorry, cannot handle strings in ensure_numeric()'
        raise Exception(msg)

    if typecode is None:
        if isinstance(A, np.ndarray):
            return A
        else:
            return np.array(A)
    else:
        return np.array(A, dtype=typecode, copy=False)


def roundoff(a, digit=2):
    """
    roundoff the number with specified digits.

    :param a: float
    :param digit:
    :return:

    :Examples:
    >>> roundoff(3.44e10, digit=2)
     3.4e10
    >>> roundoff(3.49e-10, digit=2)
     3.5e-10
    """
    if a > 1:
        return round(a, -int(math.log10(a)) + digit - 1)
    else:
        return round(a, -int(math.log10(a)) + digit)
