# _*_ coding: utf-8 _*_

# Copyright (c) 2020 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
  Perform curvature scale space in python. The method can be used to
  smooth the contour lines.
  refer to  https://github.com/makokal/pycss 
"""

import numpy as np


def _gaussian_kernel(sigma, order, t):
    """ _gaussian_kernel(sigma, order, t)
    Calculate a Gaussian kernel of the given sigma and with the given
    order, using the given t-values.
    """

    # if sigma 0, kernel is a single 1.
    if sigma == 0:
        return np.array([1.0])

    # pre-calculate some stuff
    sigma2 = sigma ** 2
    sqrt2 = np.sqrt(2)

    # Calculate the gaussian, it is unnormalized. We'll normalize at the end.
    basegauss = np.exp(- t ** 2 / (2 * sigma2))

    # Scale the t-vector, what we actually do is H( t/(sigma*sqrt2) ),
    # where H() is the Hermite polynomial.
    x = t / (sigma * sqrt2)

    # Depending on the order, calculate the Hermite polynomial already generated
    # from mathematica
    if order < 0:
        raise Exception("The order should not be negative!")
    elif order == 0:
        part = 1
    elif order == 1:
        part = 2 * x
    elif order == 2:
        part = -2 + 4 * x ** 2
    else:
        raise Exception("Order above 2 is not implemented!")

    # Apply Hermite polynomial to gauss
    k = (-1) ** order * part * basegauss

    # By calculating the normalization factor by integrating the gauss, rather
    # than using the expression 1/(sigma*sqrt(2pi)), we know that the KERNEL
    # volume is 1 when the order is 0.
    norm_default = 1 / basegauss.sum()
    #           == 1 / ( sigma * sqrt(2*pi) )

    # Here's another normalization term that we need because we use the
    # Hermite polynomials.
    norm_hermite = 1 / (sigma * sqrt2) ** order

    # Normalize and return
    return k * (norm_default * norm_hermite)


def gaussian_kernel(sigma, order=0, N=None, returnt=False):
    """ gaussian_kernel(sigma, order, N, returnt)
    Compute the gaussian kernel given a width and derivative order and optionally
    the length.
    Parameters
    -------------
    sigma : float
        Width of the Gaussian kernel
    order : int
        Derivative order of the kernel
    N : int, optional
        Number of samples to return
    returnt : Bool
        Whether or not to return the abscissa
    Returns
    -----------
    k : float
        The samples
    t : float
        Sample indices
    """

    # checking inputs
    if not N:
        # Calculate ratio that is small, but large enough to prevent errors
        ratio = 3 + 0.25 * order - 2.5 / ((order - 6) ** 2 + (order - 9) ** 2)
        # Calculate N
        N = int(np.ceil(ratio * sigma)) * 2 + 1

    elif N > 0:
        if not isinstance(N, int):
            N = int(np.ceil(N))

    elif N < 0:
        N = -N
        if not isinstance(N, int):
            N = int(np.ceil(N))
        N = N * 2 + 1

    # Check whether given sigma is large enough
    sigmaMin = 0.5 + order ** (0.62) / 5
    if sigma < sigmaMin:
        print('WARNING: The scale (sigma) is very small for the given order, '
                'better use a larger scale!')

        # Create t vector which indicates the x-position
    t = np.arange(-N / 2.0 + 0.5, N / 2.0, 1.0, dtype=np.float64)

    # Get kernel
    k = _gaussian_kernel(sigma, order, t)

    # Done
    if returnt:
        return k, t
    else:
        return k


def smooth_signal(signal, kernel):
    """ smooth_signal(signal, kernel)
    Smooth the given 1D signal by convolution with a specified kernel
    """
    return np.convolve(signal, kernel, mode='same')


def compute_curvature(curve, sigma):
    """ compute_curvature(curve, sigma)
    Compute the curvature of a 2D curve as given in Mohkatarian et. al.
    and return the curvature signal at the given sigma
    Components of the 2D curve are:
    curve[0,:] and curve[1,:]
    Parameters
    -------------
    curve : numpy matrix
        Two row matrix representing 2D curve
    sigma : float
        Kernel width
    """

    if curve[0, :].size < 2:
        raise Exception("Curve must have at least 2 points")

    sigx = curve[0, :]
    sigy = curve[1, :]
    g = gaussian_kernel(sigma, 0, sigx.size, False)
    g_s = gaussian_kernel(sigma, 1, sigx.size, False)
    g_ss = gaussian_kernel(sigma, 2, sigx.size, False)

    X_s = smooth_signal(sigx, g_s)
    Y_s = smooth_signal(sigy, g_s)
    X_ss = smooth_signal(sigx, g_ss)
    Y_ss = smooth_signal(sigy, g_ss)

    kappa = ((X_s * Y_ss) - (X_ss * Y_s)) / (X_s**2 + Y_s**2)**(1.5)

    return kappa, smooth_signal(sigx, g), smooth_signal(sigy, g)


class CurvatureScaleSpace(object):
    """ Curvature Scale Space
    A simple curvature scale space implementation based on
    Mohkatarian et. al. paper. Full algorithm detailed in
    Okal msc thesis

    :Examples:
      curve = simple_signal(np_points=400)
      c = CurvatureScaleSpace()
      cs = c.generate_css(curve, curve.shape[1], 0.01)
    """

    def __init__(self):
        pass

    def find_zero_crossings(self, kappa):
        """ find_zero_crossings(kappa)
        Locate the zero crossing points of the curvature signal kappa(t)
        """

        crossings = []

        for i in range(0, kappa.size - 2):
            if (kappa[i] < 0.0 and kappa[i + 1] > 0.0) or (kappa[i] > 0.0 and kappa[i + 1] < 0.0):
                crossings.append(i)

        return crossings

    def generate_css(self, curve, max_sigma, step_sigma):
        """ generate_css(curve, max_sigma, step_sigma)
        Generates a CSS image representation by repetatively smoothing the initial curve L_0 with increasing sigma
        """

        cols = curve[0, :].size
        rows = max_sigma // step_sigma
        css = np.zeros(shape=(rows, cols))

        srange = np.linspace(1, max_sigma - 1, rows)
        for i, sigma in enumerate(srange):
            kappa, sx, sy = compute_curvature(curve, sigma)

            # find interest points
            xs = self.find_zero_crossings(kappa)

            # save the interest points
            if len(xs) > 0 and sigma < max_sigma - 1:
                for c in xs:
                    css[i, c] = sigma  # change to any positive

            else:
                return css

    def generate_visual_css(self, rawcss, closeness, return_all=False):
        """ generate_visual_css(rawcss, closeness)
        Generate a 1D signal that can be plotted to depict the CSS by taking
        column maximums. Further checks for close interest points and nicely
        smoothes them with weighted moving average
        """

        flat_signal = np.amax(rawcss, axis=0)

        # minor smoothing via moving averages
        window = closeness
        weights = gaussian_kernel(window, 0, window, False)  # gaussian weights
        sig = np.convolve(flat_signal, weights)[window - 1:-(window - 1)]

        maxs = []

        # get maximas
        w = sig.size

        for i in range(1, w - 1):
            if sig[i - 1] < sig[i] and sig[i] > sig[i + 1]:
                maxs.append([i, sig[i]])

        if return_all:
            return sig, maxs
        else:
            return sig

    def generate_eigen_css(self, rawcss, return_all=False):
        """ generate_eigen_css(rawcss, return_all)
        Generates Eigen-CSS features
        """
        rowsum = np.sum(rawcss, axis=0)
        csum = np.sum(rawcss, axis=1)

        # hack to trim c
        colsum = csum[0:rowsum.size]

        freq = np.fft.fft(rowsum)
        mag = abs(freq)

        tilde_rowsum = np.fft.ifft(mag)

        feature = np.concatenate([tilde_rowsum, colsum], axis=0)

        if not return_all:
            return feature
        else:
            return feature, rowsum, tilde_rowsum, colsum


class SlicedCurvatureScaleSpace(CurvatureScaleSpace):
    """ Sliced Curvature Scale Space
    A implementation of the SCSS algorithm as detailed in Okal thesis
    """
    def __init__(self):
        pass

    def generate_scss(self, curves, resample_size, max_sigma, step_sigma):
        """ generate_scss
        Generate the SCSS image
        """

        scss = np.zeros(shape=(len(curves), resample_size))  # TODO - fix this hack
        # maxs = np.zeros(shape=(len(curves), resample_size))

        for i, curve in enumerate(curves):
            scss[i, :] = self.generate_css(curve, max_sigma, step_sigma)

        return scss

