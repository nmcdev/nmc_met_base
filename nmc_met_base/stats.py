# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
Statistic functions.
"""

import numpy as np


def edf(data, alpha=0.05, x0=None, x1=None, n=100):
    """
    Estimating empirical cumulative density functions (CDFs) and
      their confidence intervals from data.
    
    refer to https://james-brennan.github.io/posts/edf/

    Args:
        data (numpy.array): numpy data array.
        alpha (float, optional): [description]. Defaults to 0.05.
        x0 ([type], optional): [description]. Defaults to None.
        x1 ([type], optional): [description]. Defaults to None.
        n (int, optional): number of estimating points.

    Examples:
        import scipy.stats
        import matplotlib.pyplot as plt
        import seaborn as sns

        data = scipy.stats.gamma(5,1).rvs(200)
        x, y, l, u = edf(data, alpha=0.05)
        plt.fill_between(x, l, u)
        plt.plot(x, y, 'k-')
        plt.title("Empirical distribution function - $\hat{F}(x)$")
        plt.xlabel("$x$")
        plt.ylabel("Density")
        plt.plot(data, [0.01]*len(data), '|', color='k')
        sns.despine()
    """

    # set edf range
    x0 = data.min() if x0 is None else x0
    x1 = data.max() if x1 is None else x1

    # set estimating points
    x = np.linspace(x0, x1, n)  # estimating points

    # prepare estimating parameters
    N = data.size
    y = np.zeros_like(x)    # edf values
    l = np.zeros_like(x)    # lower confidence interval
    u = np.zeros_like(x)    # upper confidence interval

    # The Dvoretzky–Kiefer–Wolfowitz (DKW) inequality provides a method to
    # a non-parametric upper and lower confidence interval 
    e = np.sqrt(1.0/(2*N) * np.log(2./alpha))

    # calculation
    for i, xx in enumerate(x):
        y[i] = np.sum(data <= xx)/N
        l[i] = np.maximum( y[i] - e, 0 )
        u[i] = np.minimum( y[i] + e, 1 )
    return x, y, l, u


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


def lowess(x, y, f=1./3.):
    """
    Basic LOWESS smoother with uncertainty. 
    Note:
        - Not robust (so no iteration) and
             only normally distributed errors. 
        - No higher order polynomials d=1 
            so linear smoother.

    refer to: https://james-brennan.github.io/posts/lowess_conf/

    Args:
        x ([type]): [description]
        y ([type]): [description]
        f ([type], optional): [description]. Defaults to 1./3.

    Examples:
    x = 5*np.random.random(100)
    y = np.sin(x) * 3*np.exp(-x) + np.random.normal(0, 0.2, 100)

    #run it
    y_sm, y_std = lowess(x, y, f=1./5.)
    # plot it
    plt.plot(x[order], y_sm[order], color='tomato', label='LOWESS')
    plt.fill_between(x[order], y_sm[order] - y_std[order],
                     y_sm[order] + y_std[order], alpha=0.3, label='LOWESS uncertainty')
    plt.plot(x, y, 'k.', label='Observations')
    plt.legend(loc='best')
    """

    # get some paras
    xwidth = f*(x.max()-x.min())
    
    # effective width after reduction factor
    N = len(x) # number of obs
    
    # Don't assume the data is sorted
    order = np.argsort(x)
    
    # storage
    y_sm = np.zeros_like(y)
    y_stderr = np.zeros_like(y)

    # define the weigthing function -- clipping too!
    tricube = lambda d : np.clip((1- np.abs(d)**3)**3, 0, 1)

    # run the regression for each observation i
    for i in range(N):
        dist = np.abs((x[order][i]-x[order]))/xwidth
        w = tricube(dist)
        # form linear system with the weights
        A = np.stack([w, x[order]*w]).T
        b = w * y[order]
        ATA = A.T.dot(A)
        ATb = A.T.dot(b)
        # solve the syste
        sol = np.linalg.solve(ATA, ATb)
        # predict for the observation only
        yest = A[i].dot(sol)# equiv of A.dot(yest) just for k
        place = order[i]
        y_sm[place]=yest 
        sigma2 = (np.sum((A.dot(sol) -y [order])**2)/N )
        # Calculate the standard error
        y_stderr[place] = np.sqrt(sigma2 * A[i].dot(np.linalg.inv(ATA)).dot(A[i]))
    
    return y_sm, y_stderr
