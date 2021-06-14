# _*_ coding: utf-8 _*_

# Copyright (c) 2021 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
This module contains:
- a class to handle variance spectrum;
- a function to compute DCT spectrum from a 2D field;
- a function to plot a series of spectra.

refer to:
https://journals.ametsoc.org/view/journals/mwre/130/7/1520-0493_2002_130_1812_sdotda_2.0.co_2.xml
https://journals.ametsoc.org/view/journals/mwre/145/9/mwr-d-17-0056.1.xml
http://www.umr-cnrm.fr/gmapdoc/meshtml/EPYGRAM1.4.3/_modules/epygram/spectra.html
http://www.umr-cnrm.fr/gmapdoc/meshtml/EPYGRAM1.4.3/gallery/notebooks/spectral_filtering_and_spectra.html
https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/

Examples:
    variance = dctspectrum(precipitation)    # precipitation is a 2D np.array
    spectra = Spectrum(variance, name="5km Precipitation", resolution=5)
    print(spectra.wavenumbers)
    print(spectra.wavelengths)
    plotspectra(spectra)
"""


import numpy as np
import scipy.fftpack as tfm


class Spectrum(object):
    """
    A spectrum can be seen as a quantification of a signal's variance with
    regards to scale.
    If the signal is defined in physical space on N points, its spectral
    representation will be a squared mean value (wavenumber 0) and variances for
    N-1 wavenumbers.

    For details and documentation, see
        Denis et al. (2002) : 'Spectral Decomposition of Two-Dimensional
        Atmospheric Fields on Limited-Area Domains
        Using the Discrete Cosine Transform (DCT)'
    """

    def __init__(self, variances, resolution=5, name="Spectrum"):
        """
        :param variances: variances of the spectrum, from wavenumber 1 to N-1.
        :param name: an optional name for the spectrum.
        :param resolution: an optional resolution for the field represented by
                           the spectrum. It is used to compute the according
                           wavelengths. Resolution unit is arbitrary, to 
                           the will of the user.
        """
        self.variances = np.array(variances)
        self.name = name
        self.resolution = resolution

    @property
    def wavenumbers(self):
        """Gets the wavenumbers of the spectrum."""
        return np.arange(1, len(self.variances) + 1)

    @property
    def wavelengths(self):
        """Gets the wavelengths of the spectrum."""
        K = len(self.variances) + 1
        return np.array([2. * self.resolution * K / k
                         for k in self.wavenumbers])


def dctspectrum(x, verbose=False):
    """
    Function *dctspectrum* takes a 2D-array as argument and returns its 1D
    DCT ellipse spectrum.

    For details and documentation, see
        Denis et al. (2002) : 'Spectral Decomposition of Two-Dimensional
        Atmospheric Fields on Limited-Area Domains Using
        the Discrete Cosine Transform (DCT).'

    :param x: a 2D-array
    :param verbose: verbose mode
    """

    # compute transform
    if verbose:
        print("dctspectrum: compute DCT transform...")
    norm = 'ortho'  # None
    y = tfm.dct(tfm.dct(x, norm=norm, axis=0), norm=norm, axis=1)

    # compute spectrum
    if verbose:
        print("dctspectrum: compute variance spectrum...")
    N, M = y.shape
    N2 = N ** 2
    M2 = M ** 2
    MN = M * N
    K = min(M, N)
    variance = np.zeros(K)    #  variances of the spectrum, from wavenumber 1 to N-1.
    variance[0] = y[0, 0] ** 2 / MN
    for j in range(0, N):
        j2 = float(j) ** 2
        for i in range(0, M):
            var = y[j, i] ** 2 / MN
            k = np.sqrt(float(i) ** 2 / M2 + j2 / N2) * K
            k_inf = int(np.floor(k))
            k_sup = k_inf + 1
            weightsup = k - k_inf
            weightinf = 1.0 - weightsup
            if 0 <= k < 1:
                variance[1] += weightsup * var
            if 1 <= k < K - 1:
                variance[k_inf] += weightinf * var
                variance[k_sup] += weightsup * var
            if K - 1 <= k < K:
                variance[k_inf] += weightinf * var

    return variance


def plotspectra(spectra_in,
                slopes=[{'exp':-3, 'offset':1, 'label':'-3'},
                        {'exp':-5. / 3., 'offset':1, 'label':'-5/3'}],
                zoom=None, unit='SI', title=None, figsize=None):
    """
    To plot a series of spectra.

    :param spectra: a Spectrum instance or a list of.
    :param unit: string accepting LaTeX-mathematical syntaxes
    :param slopes: list of dict(
                   - exp=x where x is exposant of a A*k**-x slope
                   - offset=A where A is logscale offset in a A*k**-x slope;
                     a offset=1 is fitted to intercept the first spectra at wavenumber = 2
                   - label=(optional label) appearing 'k = label' in legend)
    :param zoom: dict(xmin=,xmax=,ymin=,ymax=)
    :param title: title for the plot
    :param figsize: figure sizes in inches, e.g. (5, 8.5).
                    Default figsize is config.plotsizes.
    """
    import matplotlib.pyplot as plt

    plt.rc('font', family='serif')
    if figsize is None:
        figsize = (14,12)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if isinstance(spectra_in, Spectrum):
        spectra = [spectra_in]
    
    # prepare dimensions
    window = dict()
    window['ymin'] = min([min(s.variances) for s in spectra]) / 10
    window['ymax'] = max([max(s.variances) for s in spectra]) * 10
    window['xmax'] = max([max(s.wavelengths) for s in spectra]) * 1.5
    window['xmin'] = min([min(s.wavelengths) for s in spectra]) * 0.8
    if zoom is not None:
        for k, v in zoom.items():
            window[k] = v
    x1 = window['xmax']
    x2 = window['xmin']

    # colors and linestyles
    colors = ['red', 'blue', 'green', 'orange', 'magenta', 'darkolivegreen',
              'yellow', 'salmon', 'black']
    linestyles = ['-', '--', '-.', ':']

    # axes
    if title is not None :
        ax.set_title(title)
    ax.set_yscale('log')
    ax.set_ylim(window['ymin'], window['ymax'])
    ax.set_xscale('log')
    ax.set_xlim(window['xmax'], window['xmin'])
    ax.grid()
    ax.set_xlabel('Wavelength ($km$)')
    ax.set_ylabel(r'Variance Spectrum ($' + unit + '$)')

    # plot slopes
    # we take the second wavenumber (of first spectrum) as intercept, because
    # it is often better fitted with spectrum than the first one
    x_intercept = spectra[0].wavelengths[1]
    y_intercept = spectra[0].variances[1]
    i = 0
    for slope in slopes:
        # a slope is defined by y = A * k**-s and we plot it with
        # two points y1, y2
        try:
            label = slope['label']
        except KeyError:
            # label = str(Fraction(slope['exp']).limit_denominator(10))
            label = str(slope['exp'])
        # because we plot w/r to wavelength, as opposed to wavenumber
        s = -slope['exp']
        A = y_intercept * x_intercept ** (-s) * slope['offset']
        y1 = A * x1 ** s
        y2 = A * x2 ** s
        ax.plot([x1, x2], [y1, y2], color='0.7',
                linestyle=linestyles[i % len(linestyles)],
                label=r'$k^{' + label + '}$')
        i += 1

    # plot spectra
    i = 0
    for s in spectra:
        ax.plot(s.wavelengths, s.variances, color=colors[i % len(colors)],
                linestyle=linestyles[i // len(colors)], label=s.name)
        i += 1

    # legend
    legend = ax.legend(loc='lower left', shadow=True)
    for label in legend.get_texts():
        label.set_fontsize('medium')

    return (fig, ax)
