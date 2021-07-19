"""
Functions to spatially interpolate data over Cartesian and spherical grids
https://github.com/tsutterley/spatial-interpolators
"""


import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
import scipy.special


def barnes_objective(xs, ys, zs, XI, YI, XR, YR, RUNS=3):
    """
    barnes_objective.py
    Written by Tyler Sutterley (08/2016)
    Barnes objective analysis for the optimal interpolation
    of an input grid using a successive corrections scheme
    CALLING SEQUENCE:
        ZI = barnes_objective(xs, ys, zs, XI, YI, XR, YR)
        ZI = barnes_objective(xs, ys, zs, XI, YI, XR, YR, RUNS=3)
    INPUTS:
        xs: input X data
        ys: input Y data
        zs: input data (Z variable)
        XI: grid X for output ZI (or array)
        YI: grid Y for output ZI (or array)
        XR: x component of Barnes smoothing length scale
            Remains fixed throughout the iterations
        YR: y component of Barnes smoothing length scale
            Remains fixed throughout the iterations
    OUTPUTS:
        ZI: interpolated grid (or array)
    OPTIONS:
        RUNS: number of iterations
    REFERENCES:
    Barnes, S. L. (1994) Applications of the Barnes objective analysis
        scheme.  Part I:  effects of undersampling, wave position, and
        station randomness.  J. of Atmos. and Oceanic Tech., 11, 1433-1448.
    Barnes, S. L. (1994) Applications of the Barnes objective analysis
        scheme.  Part II:  Improving derivative estimates.  J. of Atmos. and
        Oceanic Tech., 11, 1449-1458.
    Barnes, S. L. (1994) Applications of the Barnes objective analysis
        scheme.  Part III:  Tuning for minimum error.  J. of Atmos. and
        Oceanic Tech., 11, 1459-1479.
    Daley, R. (1991) Atmospheric data analysis, Cambridge Press, New York.
        Section 3.6.
    UPDATE HISTORY:
        Written 08/2016
    """
    #-- remove singleton dimensions
    xs = np.squeeze(xs)
    ys = np.squeeze(ys)
    zs = np.squeeze(zs)
    XI = np.squeeze(XI)
    YI = np.squeeze(YI)
    #-- size of new matrix
    if (np.ndim(XI) == 1):
        nx = len(XI)
    else:
        nx,ny = np.shape(XI)

    #-- Check to make sure sizes of input arguments are correct and consistent
    if (len(zs) != len(xs)) | (len(zs) != len(ys)):
        raise Exception('Length of X, Y, and Z must be equal')
    if (np.shape(XI) != np.shape(YI)):
        raise Exception('Size of XI and YI must be equal')

    #-- square of Barnes smoothing lengths scale
    xr2 = XR**2
    yr2 = YR**2
    #-- allocate for output zp array
    zp = np.zeros_like(XI.flatten())
    #-- first analysis
    for i,XY in enumerate(zip(XI.flatten(),YI.flatten())):
        dx = np.abs(xs - XY[0])
        dy = np.abs(ys - XY[1])
        #-- calculate weights
        w = np.exp(-dx**2/xr2 - dy**2/yr2)
        zp[i] = np.sum(zs*w)/sum(w)

    #-- allocate for even and odd zp arrays if iterating
    if (RUNS > 0):
        zpEven = np.zeros_like(zs)
        zpOdd = np.zeros_like(zs)
    #-- for each run
    for n in range(RUNS):
        #-- calculate even and odd zp arrays
        for j,xy in enumerate(zip(xs,ys)):
            dx = np.abs(xs - xy[0])
            dy = np.abs(ys - xy[1])
            #-- calculate weights
            w = np.exp(-dx**2/xr2 - dy**2/yr2)
            if ((n % 2) == 0):#-- even (% = modulus)
                zpEven[j] = zpOdd[j] + np.sum((zs - zpOdd)*w)/np.sum(w)
            else:#-- odd
                zpOdd[j] = zpEven[j] + np.sum((zs - zpEven)*w)/np.sum(w)
        #-- calculate zp for run n
        for i,XY in enumerate(zip(XI.flatten(),YI.flatten())):
            dx = np.abs(xs - XY[0])
            dy = np.abs(ys - XY[1])
            w = np.exp(-dx**2/xr2 - dy**2/yr2)
            if ((n % 2) == 0):#-- even (% = modulus)
                zp[i] = zp[i] + np.sum((zs - zpEven)*w)/np.sum(w)
            else:#-- odd
                zp[i] = zp[i] + np.sum((zs - zpOdd)*w)/np.sum(w)

    #-- reshape to original dimensions
    if (np.ndim(XI) != 1):
        ZI = zp.reshape(nx,ny)
    else:
        ZI = zp.copy()

    #-- return output matrix/array
    return ZI


def biharmonic_spline(xs, ys, zs, XI, YI, TENSION=0, REGULAR=False, EPS=1e-7):
    """
    biharmonic_spline.py
    Written by Tyler Sutterley (09/2017)
    Interpolates a sparse grid using 2D biharmonic splines (Sandwell, 1987)
    With or without tension parameters (Wessel and Bercovici, 1998)
    or using the regularized function of Mitasova and Mitas 1993
    CALLING SEQUENCE:
        ZI = biharmonic_spline(xs, ys, zs, XI, YI)
    INPUTS:
        xs: input X data
        ys: input Y data
        zs: input data (Z variable)
        XI: grid X for output ZI (or array)
        YI: grid Y for output ZI (or array)
    OUTPUTS:
        ZI: interpolated grid (or array)
    OPTIONS:
        TENSION: tension to use in interpolation (between 0 and 1)
        REGULAR: use regularized function of Mitasova and Mitas
        EPS: minimum distance value for valid points (default 1e-7)
    REFERENCES:
        Sandwell, D. T. (1987), Biharmonic spline interpolation of GEOS-3 and
            SEASAT altimeter data, Geophysical Research Letters, Vol. 2.
        Wessel and Bercovici (1998), Interpolation with Splines in Tension: A
            Green's Function Approach, Mathematical Geology, Vol. 30, No. 1.
        Mitasova and Mitas (1993), Mathematical Geology, Vol. 25, No. 6
    UPDATE HISTORY:
        Updated 09/2017: use rcond=-1 in numpy least-squares algorithms
        Updated 08/2016: detrend input data and retrend output data. calculate c
            added regularized function of Mitasova and Mitas
        Updated 06/2016: added TENSION parameter (Wessel and Bercovici, 1998)
        Written 06/2016
    """
    #-- remove singleton dimensions
    xs = np.squeeze(xs)
    ys = np.squeeze(ys)
    zs = np.squeeze(zs)
    XI = np.squeeze(XI)
    YI = np.squeeze(YI)
    #-- size of new matrix
    if (np.ndim(XI) == 1):
        nx = len(XI)
    else:
        nx,ny = np.shape(XI)

    #-- Check to make sure sizes of input arguments are correct and consistent
    if (len(zs) != len(xs)) | (len(zs) != len(ys)):
        raise Exception('Length of X, Y, and Z must be equal')
    if (np.shape(XI) != np.shape(YI)):
        raise Exception('Size of XI and YI must be equal')
    if (TENSION < 0) or (TENSION >= 1):
        raise ValueError('TENSION must be greater than 0 and less than 1')

    #-- Compute GG matrix for GG*m = d inversion problem
    npts = len(zs)
    GG = np.zeros((npts,npts))
    #-- Computation of distance Matrix (data to data)
    Rd=distance_matrix(np.array([xs, ys]),np.array([xs, ys]))
    #-- Calculate length scale for regularized case (Mitasova and Mitas)
    length_scale = np.sqrt((XI.max() - XI.min())**2 + (YI.max() - YI.min())**2)
    #-- calculate Green's function for valid points (with or without tension)
    ii,jj = np.nonzero(Rd >= EPS)
    if (TENSION == 0):
        GG[ii,jj] = (Rd[ii,jj]**2) * (np.log(Rd[ii,jj]) - 1.0)
    elif REGULAR:
        GG[ii,jj] = regular_spline2D(Rd[ii,jj], TENSION, length_scale/50.0)
    else:
        GG[ii,jj] = green_spline2D(Rd[ii,jj], TENSION)
    #-- detrend dataset
    z0,r0,p = detrend2D(xs,ys,zs)
    #-- Compute model m for detrended data
    m = np.linalg.lstsq(GG,z0,rcond=-1)[0]

    #-- Computation of distance Matrix (data to mesh points)
    Re=distance_matrix(np.array([XI.flatten(),YI.flatten()]),np.array([xs,ys]))
    gg = np.zeros_like(Re)
    #-- calculate Green's function for valid points (with or without tension)
    ii,jj = np.nonzero(Re >= EPS)
    if (TENSION == 0):
        gg[ii,jj] = (Re[ii,jj]**2) * (np.log(Re[ii,jj]) - 1.0)
    elif REGULAR:
        gg[ii,jj] = regular_spline2D(Re[ii,jj], TENSION, length_scale/50.0)
    else:
        gg[ii,jj] = green_spline2D(Re[ii,jj], TENSION)

    #-- Find 2D interpolated surface through irregular/regular X, Y grid points
    if (np.ndim(XI) == 1):
        ZI = np.squeeze(np.dot(gg,m))
    else:
        ZI = np.zeros((nx,ny))
        ZI[:,:] = np.dot(gg,m).reshape(nx,ny)
    #-- return output matrix after retrending
    return (ZI + r0[2]) + (XI-r0[0])*p[0] + (YI-r0[1])*p[1]

#-- Removing mean and slope in 2-D dataset
#-- http://www.soest.hawaii.edu/wessel/tspline/
def detrend2D(xi, yi, zi):
    #-- Find mean values
    r0 = np.zeros((3))
    r0[0] = xi.mean()
    r0[1] = yi.mean()
    r0[2] = zi.mean()
    #-- Extract mean values from X, Y and Z
    x0 = xi - r0[0]
    y0 = yi - r0[1]
    z0 = zi - r0[2]
    #-- Find slope parameters
    p = np.linalg.lstsq(np.transpose([x0,y0]),z0,rcond=-1)[0]
    #-- Extract slope from data
    z0 = z0 - x0*p[0] - y0*p[1]
    #-- return the detrended value, the mean values, and the slope parameters
    return (z0, r0, p)

#-- calculate Euclidean distances between points as matrices
def distance_matrix(x,cntrs):
    s,M = np.shape(x)
    s,N = np.shape(cntrs)
    D = np.zeros((M,N))
    for d in range(s):
        ii, = np.dot(d,np.ones((1,N))).astype(np.int)
        jj, = np.dot(d,np.ones((1,M))).astype(np.int)
        dx = x[ii,:].transpose() - cntrs[jj,:]
        D += dx**2
    D = np.sqrt(D)
    return D

#-- Green function for 2-D spline in tension (Wessel et al, 1998)
#-- http://www.soest.hawaii.edu/wessel/tspline/
def green_spline2D(x, t):
    #-- in tension: G(u) = G(u) - log(u)
    #-- where u = c * x and c = sqrt (t/(1-t))
    c = np.sqrt(t/(1.0 - t))
    #-- allocate for output Green's function
    G = np.zeros_like(x)
    #-- inverse of tension parameter
    inv_c = 1.0/c
    #-- log(2) - 0.5772156
    g0 = 0.115931515658412420677337
    #-- find points below (or equal to) 2 times inverse tension parameter
    ii, = np.nonzero(x <= (2.0*inv_c))
    u = c*x[ii]
    y = 0.25*(u**2)
    z = (u**2)/14.0625
    #-- Green's function for points ii (less than or equal to 2.0*c)
    G[ii] = (-np.log(0.5*u) * (z * (3.5156229 + z * (3.0899424 + z * \
        (1.2067492 + z * (0.2659732 + z * (0.360768e-1 + z*0.45813e-2))))))) + \
        (y * (0.42278420 + y * (0.23069756 + y * (0.3488590e-1 + \
        y * (0.262698e-2 + y * (0.10750e-3 + y * 0.74e-5))))))
    #-- find points above 2 times inverse tension parameter
    ii, = np.nonzero(x > 2.0*inv_c)
    y = 2.0*inv_c/x[ii]
    u = c*x[ii]
    #-- Green's function for points ii (greater than 2.0*c)
    G[ii] = (np.exp(-u)/np.sqrt(u)) * (1.25331414 + y * (-0.7832358e-1 + y * \
        (0.2189568e-1 + y * (-0.1062446e-1 + y * (0.587872e-2 + y * \
        (-0.251540e-2 + y * 0.53208e-3)))))) + np.log(u) - g0
    return G

#-- Regularized spline in tension (Mitasova and Mitas, 1993)
def regular_spline2D(r, t, l):
    #-- calculate tension parameter
    p = np.sqrt(t/(1.0 - t))/l
    z = (0.5 * p * r)**2
    #-- allocate for output Green's function
    G = np.zeros_like(r)
    #-- Green's function for points A (less than or equal to 1)
    A = np.nonzero(z <= 1.0)
    G[A] =  0.99999193*z[A]
    G[A] -= 0.24991055*z[A]**2
    G[A] += 0.05519968*z[A]**3
    G[A] -= 0.00976004*z[A]**4
    G[A] += 0.00107857*z[A]**4
    #-- Green's function for points B (greater than 1)
    B = np.nonzero(z > 1.0)
    En = 0.2677737343 +  8.6347608925 * z[B]
    Ed = 3.9584869228 + 21.0996530827 * z[B]
    En += 18.0590169730 * z[B]**2
    Ed += 25.6329561486 * z[B]**2
    En += 8.5733287401 * z[B]**3
    Ed += 9.5733223454 * z[B]**3
    En += z[B]**4
    Ed += z[B]**4
    G[B] = np.log(z[B]) + 0.577215664901 + (En/Ed)/(z[B]*np.exp(z[B]))
    return G


def compact_radial_basis(xs, ys, zs, XI, YI, dimension, order, smooth=0.,
    radius=None, method='wendland'):
    """
    compact_radial_basis.py
    Written by Tyler Sutterley (02/2019)
    Interpolates a sparse grid using compactly supported radial basis functions
        of minimal degree (Wendland functions) and sparse matrix algebra
        Wendland functions have the form
            p(r)    if 0 <= r <= 1
            0        if r > 1
        where p represents a univariate polynomial
    CALLING SEQUENCE:
        ZI = compact_radial_basis(xs, ys, zs, XI, YI, dimension, order,
            smooth=smooth, radius=radius, method='wendland')
    INPUTS:
        xs: scaled input X data
        ys: scaled input Y data
        zs: input data (Z variable)
        XI: scaled grid X for output ZI (or array)
        YI: scaled grid Y for output ZI (or array)
        dimension: spatial dimension of Wendland function (d)
        order: smoothness order of Wendland function (k)
    OUTPUTS:
        ZI: interpolated data grid (or array)
    OPTIONS:
        smooth: smoothing weights
        radius: scaling factor for the basis function (the radius of the
            support of the function)
        method: compactly supported radial basis function
            buhmann (not yet implemented)
            wendland (default)
            wu (not yet implemented)
    PYTHON DEPENDENCIES:
        numpy: Scientific Computing Tools For Python (https://numpy.org)
        scipy: Scientific Tools for Python (https://docs.scipy.org/doc/)
    REFERENCES:
        Holger Wendland, "Piecewise polynomial, positive definite and compactly
            supported radial functions of minimal degree." Advances in Computational
            Mathematics, 1995.
        Holger Wendland, "Scattered Data Approximation", Cambridge Monographs on
            Applied and Computational Mathematics, 2005.
        Martin Buhmann, "Radial Basis Functions", Cambridge Monographs on
            Applied and Computational Mathematics, 2003.
    UPDATE HISTORY:
        Updated 02/2019: compatibility updates for python3
        Updated 09/2017: using rcond=-1 in numpy least-squares algorithms
        Updated 08/2016: using format text within ValueError, edit constant vector
            removed 3 dimensional option of radial basis (spherical)
            changed hierarchical_radial_basis to compact_radial_basis using
                compactly-supported radial basis functions and sparse matrices
            added low-order polynomial option (previously used default constant)
        Updated 01/2016: new hierarchical_radial_basis function
            that first reduces to points within distance.  added cutoff option
        Updated 10/2014: added third dimension (spherical)
        Written 08/2014
    """
    #-- remove singleton dimensions
    xs = np.squeeze(xs)
    ys = np.squeeze(ys)
    zs = np.squeeze(zs)
    XI = np.squeeze(XI)
    YI = np.squeeze(YI)
    #-- size of new matrix
    if (np.ndim(XI) == 1):
        nx = len(XI)
    else:
        nx,ny = np.shape(XI)

    #-- Check to make sure sizes of input arguments are correct and consistent
    if (len(zs) != len(xs)) | (len(zs) != len(ys)):
        raise Exception('Length of X, Y, and Z must be equal')
    if (np.shape(XI) != np.shape(YI)):
        raise Exception('Size of XI and YI must be equal')

    #-- create python dictionary of compact radial basis function formulas
    radial_basis_functions = {}
    # radial_basis_functions['buhmann'] = buhmann
    radial_basis_functions['wendland'] = wendland
    # radial_basis_functions['wu'] = wu
    #-- check if formula name is listed
    if method in radial_basis_functions.keys():
        cRBF = radial_basis_functions[method]
    else:
        raise ValueError("Method {0} not implemented".format(method))

    #-- construct kd-tree for Data points
    kdtree = scipy.spatial.cKDTree(list(zip(xs, ys)))
    if radius is None:
        #-- quick nearest-neighbor lookup to calculate mean radius
        ds,_ = kdtree.query(list(zip(xs, ys)), k=2)
        radius = 2.0*np.mean(ds[:, 1])

    #-- Creation of data-data distance sparse matrix in COOrdinate format
    Rd = kdtree.sparse_distance_matrix(kdtree, radius, output_type='coo_matrix')
    #-- calculate ratio between data-data distance and radius
    #-- replace cases where the data-data distance is greater than the radius
    r0 = np.where(Rd.data < radius, Rd.data/radius, radius/radius)
    #-- calculation of model PHI
    PHI = cRBF(r0, dimension, order)
    #-- construct sparse radial matrix
    PHI = scipy.sparse.coo_matrix((PHI, (Rd.row,Rd.col)), shape=Rd.shape)
    #-- Augmentation of the PHI Matrix with a smoothing factor
    if (smooth != 0):
        #-- calculate eigenvalues of distance matrix
        eig = scipy.sparse.linalg.eigsh(Rd, k=1, which="LA", maxiter=1000,
            return_eigenvectors=False)[0]
        PHI += scipy.sparse.identity(len(xs), format='coo') * smooth * eig

    #-- Computation of the Weights
    w = scipy.sparse.linalg.spsolve(PHI, zs)

    #-- construct kd-tree for Mesh points
    #-- Data to Mesh Points
    mkdtree = scipy.spatial.cKDTree(list(zip(XI.flatten(),YI.flatten())))
    #-- Creation of data-mesh distance sparse matrix in COOrdinate format
    Re = kdtree.sparse_distance_matrix(mkdtree,radius,output_type='coo_matrix')
    #-- calculate ratio between data-mesh distance and radius
    #-- replace cases where the data-mesh distance is greater than the radius
    R0 = np.where(Re.data < radius, Re.data/radius, radius/radius)
    #-- calculation of the Evaluation Matrix
    E = cRBF(R0, dimension, order)
    #-- construct sparse radial matrix
    E = scipy.sparse.coo_matrix((E, (Re.row,Re.col)), shape=Re.shape)

    #-- calculate output interpolated array (or matrix)
    if (np.ndim(XI) == 1):
        ZI = E.transpose().dot(w[:,np.newaxis])
    else:
        ZI = np.zeros((nx,ny))
        ZI[:,:] = E.transpose().dot(w[:,np.newaxis]).reshape(nx,ny)
    #-- return the interpolated array (or matrix)
    return ZI

#-- define compactly supported radial basis function formulas
def wendland(r,d,k):
    #-- Wendland functions of dimension d and order k
    #-- can replace with recursive method of Wendland for generalized case
    L = (d//2) + k + 1
    if (k == 0):
        f = (1. - r)**L
    elif (k == 1):
        f = (1. - r)**(L + 1)*((L + 1.)*r + 1.)
    elif (k == 2):
        f = (1. - r)**(L + 2)*((L**2 + 4.*L + 3.)*r**2 + (3.*L + 6.)*r + 3.)
    elif (k == 3):
        f = (1. - r)**(L + 3)*((L**3 + 9.*L**2 + 23.*L + 15.)*r**3 +
            (6.*L**2 + 36.*L + 45.)*r**2 + (15.*L + 45.)*r + 15.)
    return f


def legendre(l,x,NORMALIZE=False):
    """
    Written by Tyler Sutterley (02/2021)
    Computes associated Legendre functions of degree l evaluated for elements x
    l must be a scalar integer and x must contain real values ranging -1 <= x <= 1
    Parallels the MATLAB legendre function
    Based on Fortran program by Robert L. Parker, Scripps Institution of
    Oceanography, Institute for Geophysics and Planetary Physics, UCSD. 1993
    INPUTS:
        l: degree of Legrendre polynomials
        x: elements ranging from -1 to 1
            typically cos(theta), where theta is the colatitude in radians
    OUTPUT:
        Pl: legendre polynomials of degree l for orders 0 to l
    OPTIONS:
        NORMALIZE: output Fully Normalized Associated Legendre Functions
    PYTHON DEPENDENCIES:
        numpy: Scientific Computing Tools For Python (https://numpy.org)
        scipy: Scientific Tools for Python (https://docs.scipy.org/doc/)
    REFERENCES:
        M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
            Dover Publications, 1965, Ch. 8.
        J. A. Jacobs, "Geomagnetism", Academic Press, 1987, Ch.4.
    UPDATE HISTORY:
        Updated 02/2021: modify case with underflow
        Updated 09/2020: verify dimensions of x variable
        Updated 07/2020: added function docstrings
        Updated 05/2020: added normalization option for output polynomials
        Updated 03/2019: calculate twocot separately to avoid divide warning
        Written 08/2016

    Computes associated Legendre functions of degree l
    Arguments
    ---------
    l: degree of Legrendre polynomials
    x: elements ranging from -1 to 1
    Keyword arguments
    -----------------
    NORMALIZE: output Fully-normalized Associated Legendre Functions
    Returns
    -------
    Pl: legendre polynomials of degree l for orders 0 to l
    """
    #-- verify dimensions
    x = np.atleast_1d(x).flatten()
    #-- size of the x array
    nx = len(x)

    #-- for the l = 0 case
    if (l == 0):
        Pl = np.ones((1,nx), dtype=np.float)
        return Pl

    #-- for all other degrees greater than 0
    rootl = np.sqrt(np.arange(0,2*l+1))#-- +1 to include 2*l
    #-- s is sine of colatitude (cosine of latitude) so that 0 <= s <= 1
    s = np.sqrt(1.0 - x**2)#-- for x=cos(th): s=sin(th)
    P = np.zeros((l+3,nx), dtype=np.float)

    #-- Find values of x,s for which there will be underflow
    sn = (-s)**l
    tol = np.sqrt(np.finfo(float).tiny)
    count = np.count_nonzero((s > 0) & (np.abs(sn) <= tol))
    if (count > 0):
        ind, = np.nonzero((s > 0) & (np.abs(sn) <= tol))
        #-- Approximate solution of x*ln(x) = Pl
        v = 9.2 - np.log(tol)/(l*s[ind])
        w = 1.0/np.log(v)
        m1 = 1+l*s[ind]*v*w*(1.0058+ w*(3.819 - w*12.173))
        m1 = np.where(l < np.floor(m1), l, np.floor(m1)).astype(np.int)
        #-- Column-by-column recursion
        for k,mm1 in enumerate(m1):
            col = ind[k]
            #-- Calculate twocot for underflow case
            twocot = -2.0*x[col]/s[col]
            P[mm1-1:l+1,col] = 0.0
            #-- Start recursion with proper sign
            tstart = np.finfo(np.float).eps
            P[mm1-1,col] = np.sign(np.fmod(mm1,2)-0.5)*tstart
            if (x[col] < 0):
                P[mm1-1,col] = np.sign(np.fmod(l+1,2)-0.5)*tstart
            #-- Recur from m1 to m = 0, accumulating normalizing factor.
            sumsq = tol.copy()
            for m in range(mm1-2,-1,-1):
                P[m,col] = ((m+1)*twocot*P[m+1,col] - \
                    rootl[l+m+2]*rootl[l-m-1]*P[m+2,col]) / \
                    (rootl[l+m+1]*rootl[l-m])
                sumsq += P[m,col]**2
            #-- calculate scale
            scale = 1.0/np.sqrt(2.0*sumsq - P[0,col]**2)
            P[0:mm1+1,col] = scale*P[0:mm1+1,col]

    #-- Find the values of x,s for which there is no underflow, and (x != +/-1)
    count = np.count_nonzero((x != 1) & (np.abs(sn) >= tol))
    if (count > 0):
        nind, = np.nonzero((x != 1) & (np.abs(sn) >= tol))
        #-- Calculate twocot for normal case
        twocot = -2.0*x[nind]/s[nind]
        #-- Produce normalization constant for the m = l function
        d = np.arange(2,2*l+2,2)
        c = np.prod(1.0 - 1.0/d)
        #-- Use sn = (-s)**l (written above) to write the m = l function
        P[l,nind] = np.sqrt(c)*sn[nind]
        P[l-1,nind] = P[l,nind]*twocot*l/rootl[-1]

        #-- Recur downwards to m = 0
        for m in range(l-2,-1,-1):
            P[m,nind] = (P[m+1,nind]*twocot*(m+1) - \
                P[m+2,nind]*rootl[l+m+2]*rootl[l-m-1]) / \
                (rootl[l+m+1]*rootl[l-m])

    #-- calculate Pl from P
    Pl = np.copy(P[0:l+1,:])

    #-- Polar argument (x == +/-1)
    count = np.count_nonzero(s == 0)
    if (count > 0):
        s0, = np.nonzero(s == 0)
        Pl[0,s0] = x[s0]**l

    #-- Calculate the unnormalized Legendre functions by multiplying each row
    #-- by: sqrt((l+m)!/(l-m)!) == sqrt(prod(n-m+1:n+m))
    #-- following Abramowitz and Stegun
    for m in range(1,l):
        Pl[m,:] = np.prod(rootl[l-m+1:l+m+1])*Pl[m,:]

    #-- sectoral case (l = m) should be done separately to handle 0!
    Pl[l,:] = np.prod(rootl[1:])*Pl[l,:]

    #-- calculate Fully Normalized Associated Legendre functions
    if NORMALIZE:
        norm = np.zeros((l+1))
        norm[0] = np.sqrt(2.0*l+1)
        m = np.arange(1,l+1)
        norm[1:] = (-1)**m*np.sqrt(2.0*(2.0*l+1.0)*scipy.special.factorial(l-m)/
            scipy.special.factorial(l+m))
        Pl *= np.kron(np.ones((1,nx)), norm[:,np.newaxis])

    return Pl


def radial_basis(xs, ys, zs, XI, YI, smooth=0., epsilon=None, method='inverse',
    polynomial=None):
    """
    radial_basis.py
    Written by Tyler Sutterley (09/2017)
    Interpolates a sparse grid using radial basis functions
    CALLING SEQUENCE:
        ZI = radial_basis(xs, ys, zs, XI, YI, polynomial=0,
            smooth=smooth, epsilon=epsilon, method='inverse')
    INPUTS:
        xs: scaled input X data
        ys: scaled input Y data
        data: input data (Z variable)
        XI: scaled grid X for output ZI (or array)
        YI: scaled grid Y for output ZI (or array)
    OUTPUTS:
        ZI: interpolated data grid (or array)
    OPTIONS:
        smooth: smoothing weights
        epsilon: norm input
            default is mean Euclidean distance
        polynomial: polynomial order if augmenting radial basis functions
            default None: no polynomials
        method: radial basis function
            multiquadric
            inverse_multiquadric or inverse (default)
            inverse_quadratic
            gaussian
            linear (first-order polyharmonic spline)
            cubic (third-order polyharmonic spline)
            quintic (fifth-order polyharmonic spline)
            thin_plate: thin-plate spline
    PYTHON DEPENDENCIES:
        numpy: Scientific Computing Tools For Python (https://numpy.org)
    REFERENCES:
        R. L. Hardy, Multiquadric equations of topography and other irregular
            surfaces, J. Geophys. Res., 76(8), 1905-1915, 1971.
        M. Buhmann, "Radial Basis Functions", Cambridge Monographs on Applied and
            Computational Mathematics, 2003.
    UPDATE HISTORY:
        Updated 09/2017: using rcond=-1 in numpy least-squares algorithms
        Updated 01/2017: epsilon in polyharmonic splines (linear, cubic, quintic)
        Updated 08/2016: using format text within ValueError, edit constant vector
            removed 3 dimensional option of radial basis (spherical)
            changed hierarchical_radial_basis to compact_radial_basis using
                compactly-supported radial basis functions and sparse matrices
            added low-order polynomial option (previously used default constant)
        Updated 01/2016: new hierarchical_radial_basis function
            that first reduces to points within distance.  added cutoff option
        Updated 10/2014: added third dimension (spherical)
        Written 08/2014
    """

    #-- remove singleton dimensions
    xs = np.squeeze(xs)
    ys = np.squeeze(ys)
    zs = np.squeeze(zs)
    XI = np.squeeze(XI)
    YI = np.squeeze(YI)
    #-- size of new matrix
    if (np.ndim(XI) == 1):
        nx = len(XI)
    else:
        nx,ny = np.shape(XI)

    #-- Check to make sure sizes of input arguments are correct and consistent
    if (len(zs) != len(xs)) | (len(zs) != len(ys)):
        raise Exception('Length of X, Y, and Z must be equal')
    if (np.shape(XI) != np.shape(YI)):
        raise Exception('Size of XI and YI must be equal')

    #-- create python dictionary of radial basis function formulas
    radial_basis_functions = {}
    radial_basis_functions['multiquadric'] = multiquadric
    radial_basis_functions['inverse_multiquadric'] = inverse_multiquadric
    radial_basis_functions['inverse'] = inverse_multiquadric
    radial_basis_functions['inverse_quadratic'] = inverse_quadratic
    radial_basis_functions['gaussian'] = gaussian
    radial_basis_functions['linear'] = poly_spline1
    radial_basis_functions['cubic'] = poly_spline3
    radial_basis_functions['quintic'] = poly_spline5
    radial_basis_functions['thin_plate'] = thin_plate
    #-- check if formula name is listed
    if method in radial_basis_functions.keys():
        RBF = radial_basis_functions[method]
    else:
        raise ValueError("Method {0} not implemented".format(method))

    #-- Creation of data distance matrix
    #-- Data to Data
    Rd = distance_matrix(np.array([xs, ys]),np.array([xs, ys]))
    N,M = np.shape(Rd)
    #-- if epsilon is not specified
    if epsilon is None:
        #-- calculate norm with mean euclidean distance
        uix,uiy = np.nonzero(np.tri(N,M=M,k=-1))
        epsilon = np.mean(Rd[uix,uiy])

    #-- possible augmentation of the PHI Matrix with polynomial Vectors
    if polynomial is None:
        #-- calculate radial basis function for data-to-data with smoothing
        PHI = RBF(epsilon, Rd) + np.eye(N,M=M)*smooth
        DMAT = zs.copy()
    else:
        #-- number of polynomial coefficients
        nt = (polynomial**2 + 3*polynomial)//2 + 1
        #-- calculate radial basis function for data-to-data with smoothing
        PHI = np.zeros((N+nt,M+nt))
        PHI[:N,:M] = RBF(epsilon, Rd) + np.eye(N,M=M)*smooth
        #-- augmentation of PHI matrix with polynomials
        POLY = polynomial_matrix(xs,ys,polynomial)
        DMAT = np.concatenate(([zs,np.zeros((nt))]),axis=0)
        #-- augment PHI matrix
        for t in range(nt):
            PHI[:N,M+t] = POLY[:,t]
            PHI[N+t,:M] = POLY[:,t]

    #-- Computation of the Weights
    w = np.linalg.lstsq(PHI,DMAT[:,np.newaxis],rcond=-1)[0]

    #-- Computation of distance Matrix
    #-- Data to Mesh Points
    Re=distance_matrix(np.array([XI.flatten(),YI.flatten()]),np.array([xs,ys]))
    #-- calculate radial basis function for data-to-mesh matrix
    E = RBF(epsilon,Re)

    #-- possible augmentation of the Evaluation Matrix with polynomial vectors
    if polynomial is not None:
        P = polynomial_matrix(XI.flatten(),YI.flatten(),polynomial)
        E = np.concatenate(([E, P]),axis=1)
    #-- calculate output interpolated array (or matrix)
    if (np.ndim(XI) == 1):
        ZI = np.squeeze(np.dot(E,w))
    else:
        ZI = np.zeros((nx,ny))
        ZI[:,:] = np.dot(E,w).reshape(nx,ny)
    #-- return the interpolated array (or matrix)
    return ZI

#-- define radial basis function formulas
def multiquadric(epsilon, r):
    #-- multiquadratic
    f = np.sqrt((epsilon*r)**2 + 1.0)
    return f

def inverse_multiquadric(epsilon, r):
    #-- inverse multiquadratic
    f = 1.0/np.sqrt((epsilon*r)**2 + 1.0)
    return f

def inverse_quadratic(epsilon, r):
    #-- inverse quadratic
    f = 1.0/(1.0+(epsilon*r)**2)
    return f

def gaussian(epsilon, r):
    #-- gaussian
    f = np.exp(-(epsilon*r)**2)
    return f

def poly_spline1(epsilon, r):
    #-- First-order polyharmonic spline
    f = (epsilon*r)
    return f

def poly_spline3(epsilon, r):
    #-- Third-order polyharmonic spline
    f = (epsilon*r)**3
    return f

def poly_spline5(epsilon, r):
    #-- Fifth-order polyharmonic spline
    f = (epsilon*r)**5
    return f

def thin_plate(epsilon, r):
    #-- thin plate spline
    f = r**2 * np.log(r)
    #-- the spline is zero at zero
    f[r == 0] = 0.0
    return f

#-- calculate polynomial matrix to augment radial basis functions
def polynomial_matrix(x,y,order):
    c = 0
    M = len(x)
    N = (order**2 + 3*order)//2 + 1
    POLY = np.zeros((M,N))
    for ii in range(order + 1):
        for jj in range(ii + 1):
            POLY[:,c] = (x**jj)*(y**(ii-jj))
            c += 1
    return POLY


def sph2xyz(lon,lat,RAD=6371.0):
    """
    sph2xyz.py
    Written by Tyler Sutterley (07/2013)
    Converts spherical coordinates to Cartesian coordinates
    CALLING SEQUENCE:
        xyz = sph2xyz(lon,lat,RAD=6371.0)
        x = xyz['x']
        y = xyz['y']
        z = xyz['z']
    INPUTS
        lon: spherical longitude
        lat: spherical latitude
    OUTPUTS:
        x,y,z in cartesian coordinates
    OPTIONS:
        RAD: radius (default is mean Earth radius)
    PYTHON DEPENDENCIES:
        numpy: Scientific Computing Tools For Python (https://numpy.org)
    """
    import numpy as np

    ilon = np.nonzero(lon < 0)
    count = np.count_nonzero(lon < 0)
    if (count != 0):
        lon[ilon] = lon[ilon]+360.0

    phi = np.pi*lon/180.0
    th = np.pi*(90.0 - lat)/180.0

    x=RAD*np.sin(th)*np.cos(phi)#-- x
    y=RAD*np.sin(th)*np.sin(phi)#-- y
    z=RAD*np.cos(th)#-- z

    return {'x':x,'y':y,'z':z}

def sph_bilinear(x, y, z, xi, yi, FLATTENED=False, FILL_VALUE=-9999.0):
    u"""
    sph_bilinear.py
    Written by Tyler Sutterley (09/2017)
    Spherical interpolation routine for gridded data using bilinear interpolation
    CALLING SEQUENCE:
        zi = sph_bilinear(x, y, z, xi, yi)
    INPUTS:
        x: input longitude
        y: input latitude
        z: input data (matrix)
        xi: output longitude
        yi: output latitude
    OUTPUTS:
        zi: data regridded to new global grid (or regional if using FLATTENED)
    OPTIONS:
        FLATTENED: input xi, yi are flattened arrays (nlon must equal nlat)
        FILL_VALUE: value to use if xi and yi are out of range
    PYTHON DEPENDENCIES:
        numpy: Scientific Computing Tools For Python (https://numpy.org)
    UPDATE HISTORY:
        Updated 09/2017: use minimum distances with FLATTENED method
            if indices are out of range: replace with FILL_VALUE
        Updated 03/2016: added FLATTENED option for regional grids to global grids
        Updated 11/2015: made easier to read with data and weight values
        Written 07/2013
    """
    #-- Converting input data into geodetic coordinates in radians
    phi = x*np.pi/180.0
    th = (90.0 -y)*np.pi/180.0
    #-- grid steps for lon and lat
    dlon = np.abs(x[1]-x[0])
    dlat = np.abs(y[1]-y[0])
    #-- grid steps in radians
    dphi = dlon*np.pi/180.0
    dth = dlat*np.pi/180.0
    #-- input data shape
    nx = len(x)
    ny = len(y)
    #-- Converting output data into geodetic coordinates in radians
    xphi = xi*np.pi/180.0
    xth = (90.0 -yi)*np.pi/180.0
    #-- check if using flattened array or two-dimensional lat/lon
    if FLATTENED:
        #-- output array
        ndat = len(xi)
        zi = np.zeros((ndat))
        for i in range(0,ndat):
            #-- calculating the indices for the original grid
            dx = (x - np.floor(xi[i]/dlon)*dlon)**2
            dy = (y - np.floor(yi[i]/dlat)*dlat)**2
            iph = np.argmin(dx)
            ith = np.argmin(dy)
            #-- data is within range of values
            if ((iph+1) < nx) & ((ith+1) < ny):
                #-- corner data values for i,j
                Ia = z[iph,ith]#-- (0,0)
                Ib = z[iph+1,ith]#-- (1,0)
                Ic = z[iph,ith+1]#-- (0,1)
                Id = z[iph+1,ith+1]#-- (1,1)
                #-- corner weight values for i,j
                Wa = (xphi[i]-phi[iph])*(xth[i]-th[ith])
                Wb = (phi[iph+1]-xphi[i])*(xth[i]-th[ith])
                Wc = (xphi[i]-phi[iph])*(th[ith+1]-xth[i])
                Wd = (phi[iph+1]-xphi[i])*(th[ith+1]-xth[i])
                #-- divisor weight value
                W = (phi[iph+1]-phi[iph])*(th[ith+1]-th[ith])
                #-- calculate interpolated value for i
                zi[i] = (Ia*Wa + Ib*Wb + Ic*Wc + Id*Wd)/W
            else:
                #-- replace with fill value
                zi[i] = FILL_VALUE
    else:
        #-- output grid
        nphi = len(xi)
        nth = len(yi)
        zi = np.zeros((nphi,nth))
        for i in range(0,nphi):
            for j in range(0,nth):
                #-- calculating the indices for the original grid
                iph = np.floor(xphi[i]/dphi)
                jth = np.floor(xth[j]/dth)
                #-- data is within range of values
                if ((iph+1) < nx) & ((jth+1) < ny):
                    #-- corner data values for i,j
                    Ia = z[iph,jth]#-- (0,0)
                    Ib = z[iph+1,jth]#-- (1,0)
                    Ic = z[iph,jth+1]#-- (0,1)
                    Id = z[iph+1,jth+1]#-- (1,1)
                    #-- corner weight values for i,j
                    Wa = (xphi[i]-phi[iph])*(xth[j]-th[jth])
                    Wb = (phi[iph+1]-xphi[i])*(xth[j]-th[jth])
                    Wc = (xphi[i]-phi[iph])*(th[jth+1]-xth[j])
                    Wd = (phi[iph+1]-xphi[i])*(th[jth+1]-xth[j])
                    #-- divisor weight value
                    W = (phi[iph+1]-phi[iph])*(th[jth+1]-th[jth])
                    #-- calculate interpolated value for i,j
                    zi[i,j] = (Ia*Wa + Ib*Wb + Ic*Wc + Id*Wd)/W
                else:
                    #-- replace with fill value
                    zi[i,j] = FILL_VALUE

    #-- return the interpolated data
    return zi


def sph_radial_basis(lon, lat, data, LONGITUDE, LATITUDE, smooth=0.,
    epsilon=None, method='inverse', QR=False, norm='Euclidean'):
    u"""
    sph_radial_basis.py
    Written by Tyler Sutterley (02/2019)
    Interpolates a sparse grid over a sphere using radial basis functions with
        QR factorization option to eliminate ill-conditioning (Fornberg, 2007)
    CALLING SEQUENCE:
        DATA = sph_radial_basis(lon, lat, data, LONGITUDE, LATITUDE,
            smooth=smooth, epsilon=epsilon, method='inverse')
    INPUTS:
        lon: input longitude
        lat: input latitude
        data: input data (Z variable)
        LONGITUDE: output longitude (array or grid)
        LATITUDE: output latitude (array or grid)
    OUTPUTS:
        DATA: interpolated data (array or grid)
    OPTIONS:
        smooth: smoothing weights
        epsilon: norm input
            default is mean Euclidean distance
        method: radial basis function (** has option for QR factorization method)
            multiquadric**
            inverse_multiquadric** or inverse** (default)
            inverse_quadratic**
            gaussian**
            linear
            cubic
            quintic
            thin_plate: thin-plate spline
        QR: use QR factorization algorithm of Fornberg (2007)
        norm: distance function for radial basis functions (if not using QR)
            Euclidean: Euclidean Distance with distance_matrix (default)
            GCD: Great-Circle Distance using n-vectors with angle_matrix
    PYTHON DEPENDENCIES:
        numpy: Scientific Computing Tools For Python (https://numpy.org)
        scipy: Scientific Tools for Python (https://docs.scipy.org/doc/)
    REFERENCES:
        B Fornberg and C Piret, "A stable algorithm for flat radial basis functions
            on a sphere." SIAM J. Sci. Comput. 30(1), 60-80 (2007)
        B Fornberg, E Larsson, and N Flyer, "Stable Computations with Gaussian
            Radial Basis Functions." SIAM J. Sci. Comput. 33(2), 869-892 (2011)
    UPDATE HISTORY:
        Updated 02/2019: compatibility updates for python3
        Updated 09/2017: using rcond=-1 in numpy least-squares algorithms
        Updated 08/2016: finished QR factorization method, added norm option
        Forked 08/2016 from radial_basis.py for use over a sphere
        Updated 08/2016: using format text within ValueError, edit constant vector
            removed 3 dimensional option of radial basis (spherical)
            changed hierarchical_radial_basis to compact_radial_basis using
                compactly-supported radial basis functions and sparse matrices
            added low-order polynomial option (previously used default constant)
        Updated 01/2016: new hierarchical_radial_basis function
            that first reduces to points within distance.  added cutoff option
        Updated 10/2014: added third dimension (spherical)
        Written 08/2014
    """
    #-- remove singleton dimensions
    lon = np.squeeze(lon)
    lat = np.squeeze(lat)
    data = np.squeeze(data)
    LONGITUDE = np.squeeze(LONGITUDE)
    LATITUDE = np.squeeze(LATITUDE)
    #-- size of new matrix
    if (np.ndim(LONGITUDE) > 1):
        nlon,nlat = np.shape(LONGITUDE)
        sz = np.int(nlon*nlat)
    else:
        sz = len(LONGITUDE)

    #-- Check to make sure sizes of input arguments are correct and consistent
    if (len(data) != len(lon)) | (len(data) != len(lat)):
        raise Exception('Length of Longitude, Latitude, and Data must be equal')
    if (np.shape(LONGITUDE) != np.shape(LATITUDE)):
        raise Exception('Size of output Longitude and Latitude must be equal')

    #-- create python dictionary of radial basis function formulas
    radial_basis_functions = {}
    radial_basis_functions['multiquadric'] = multiquadric
    radial_basis_functions['inverse_multiquadric'] = inverse_multiquadric
    radial_basis_functions['inverse'] = inverse_multiquadric
    radial_basis_functions['inverse_quadratic'] = inverse_quadratic
    radial_basis_functions['gaussian'] = gaussian
    radial_basis_functions['linear'] = linear
    radial_basis_functions['cubic'] = cubic
    radial_basis_functions['quintic'] = quintic
    radial_basis_functions['thin_plate'] = thin_plate
    #-- create python dictionary of radial basis function expansions
    radial_expansions = {}
    radial_expansions['multiquadric'] = multiquadratic_expansion
    radial_expansions['inverse_multiquadric'] = inverse_multiquadric_expansion
    radial_expansions['inverse'] = inverse_multiquadric_expansion
    radial_expansions['inverse_quadratic'] = inverse_quadratic_expansion
    radial_expansions['gaussian'] = gaussian_expansion
    #-- check if formula name is listed
    if method in radial_basis_functions.keys():
        RBF = radial_basis_functions[method]
    else:
        raise ValueError("Method {0} not implemented".format(method))
    #-- check if formula name is valid for QR factorization method
    if QR and (method in radial_expansions.keys()):
        expansion = radial_expansions[method]
    elif QR and (method not in radial_expansions.keys()):
        raise ValueError("{0} expansion not available with QR".format(method))
    #-- create python dictionary of distance functions (if not using QR)
    norm_functions = {}
    norm_functions['Euclidean'] = distance_matrix
    norm_functions['GCD'] = angle_matrix
    if norm in norm_functions:
        norm_matrix = norm_functions[norm]
    else:
        raise ValueError("Distance Function {0} not implemented".format(norm))

    #-- convert input lat and lon into cartesian X,Y,Z over unit sphere
    phi = np.pi*lon/180.0
    th = np.pi*(90.0 - lat)/180.0
    xs = np.sin(th)*np.cos(phi)
    ys = np.sin(th)*np.sin(phi)
    zs = np.cos(th)
    #-- convert output longitude and latitude into cartesian X,Y,Z
    PHI = np.pi*LONGITUDE.flatten()/180.0
    THETA = np.pi*(90.0 - LATITUDE.flatten())/180.0
    XI = np.sin(THETA)*np.cos(PHI)
    YI = np.sin(THETA)*np.sin(PHI)
    ZI = np.cos(THETA)

    #-- Creation of data distance matrix (Euclidean or Great-Circle Distance)
    #-- Data to Data
    Rd = norm_matrix(np.array([xs, ys, zs]),np.array([xs, ys, zs]))
    N,M = np.shape(Rd)
    #-- if epsilon is not specified
    if epsilon is None:
        #-- calculate norm with mean distance
        uix,uiy = np.nonzero(np.tri(N,M=M,k=-1))
        epsilon = np.mean(Rd[uix,uiy])

    #-- QR factorization algorithm of Fornberg (2007)
    if QR:
        #-- calculate radial basis functions using spherical harmonics
        R,w = RBF_QR(th,phi,epsilon,data,expansion)
        n_harm = np.sqrt(np.shape(R)[0]).astype(np.int)
        #-- counter variable for filling spherical harmonic matrix
        index = 0
        #-- evaluation matrix E
        E = np.zeros((sz,np.int(n_harm**2)))
        for l in range(0,n_harm):
            #-- Each loop adds a block of columns of degree l to E
            E[:,index:2*l+index+1] = spherical_harmonic_matrix(l,THETA,PHI)
            index += 2*l + 1
        #-- calculate output interpolated array (or matrix)
        DATA = np.dot(E,np.dot(R,w))
    else:
        #-- Calculation of the PHI Matrix with smoothing
        PHI = np.zeros((N+1,M+1))
        PHI[:N,:M] = RBF(epsilon, Rd) + np.eye(N,M=M)*smooth
        #-- Augmentation of the PHI Matrix with a Constant Vector
        PHI[:N,M] = np.ones((N))
        PHI[N,:M] = np.ones((M))

        #-- Computation of the Weights
        DMAT = np.concatenate(([data,[0]]),axis=0)
        w = np.linalg.lstsq(PHI,DMAT[:,np.newaxis],rcond=-1)[0]

        #-- Computation of distance Matrix (Euclidean or Great-Circle Distance)
        #-- Data to Mesh Points
        Re = norm_matrix(np.array([XI,YI,ZI]),np.array([xs,ys,zs]))
        #-- calculate radial basis function for data-to-mesh matrix
        E = RBF(epsilon,Re)

        #-- Augmentation of the Evaluation Matrix with a Constant Vector
        P = np.ones((sz,1))
        E = np.concatenate(([E, P]),axis=1)
        #-- calculate output interpolated array (or matrix)
        DATA = np.dot(E,w)

    #-- reshape output to original dimensions and return
    if (np.ndim(LONGITUDE) == 1):
        return np.squeeze(DATA)
    else:
        return DATA.reshape(nlon,nlat)

#-- define radial basis function formulas
def multiquadratic_expansion(epsilon, mu):
    c = -2.*np.pi*(2.*epsilon**2+1.+(mu+1./2.)*np.sqrt(1.+4.*epsilon**2)) / \
        (mu + 1.0/2.0)/(mu + 3.0/2.0)/(mu - 1.0/2.0) * \
        (2.0/(1.0 + np.sqrt(4.0*epsilon**2+1.0)))**(2.0*mu+1.0)
    return c

def inverse_multiquadric_expansion(epsilon, mu):
    c = 4.0*np.pi/(mu+1.0/2.0)*(2.0/(1.0+np.sqrt(4.0*epsilon**2+1.)))**(2*mu+1.)
    return c

def inverse_quadratic_expansion(epsilon, mu):
    c = 4.0*np.pi**(3.0/2.0)*scipy.special.factorial(mu) / \
        scipy.special.gamma(mu + 3.0/2.0)/(1.0 + 4.0*epsilon**2)**(mu+1) * \
        scipy.special.hyp2f1(mu+1, mu+1, 2.*mu+2, 4.*epsilon**2/(1.+4.*epsilon**2))
    return c

def gaussian_expansion(epsilon, mu):
    c = 4.0*np.pi**(3.0/2.0)*np.exp(-2.0*epsilon**2) * \
        scipy.special.iv(mu + 1.0/2.0, 2.0*epsilon**2)/epsilon**(2.0*mu + 1.0)
    return c

def linear(epsilon, r):
    #-- linear polynomial
    return r

def cubic(epsilon, r):
    #-- cubic polynomial
    f = r**3
    return f

def quintic(epsilon, r):
    #-- quintic polynomial
    f = r**5
    return f

#-- calculate great-circle distance between between n-vectors
def angle_matrix(x,cntrs):
    _,M = np.shape(x)
    _,N = np.shape(cntrs)
    A = np.zeros((M,N))
    A[:,:] = np.arccos(np.dot(x.transpose(), cntrs))
    A[np.isnan(A)] = 0.0
    return A

#-- calculate spherical harmonics of degree l evaluated at (theta,phi)
def spherical_harmonic_matrix(l,theta,phi):
    #-- calculate legendre polynomials
    nth = len(theta)
    Pl = legendre(l,np.cos(theta)).transpose()
    #-- calculate degree dependent factors C and F
    m = np.arange(0,l+1)#-- spherical harmonic orders up to degree l
    C = np.sqrt((2.0*l + 1.0)/(4.0*np.pi))
    F=np.sqrt(scipy.special.factorial(1+l-m-1)/scipy.special.factorial(1+l+m-1))
    F=np.kron(np.ones((nth,1)), F[np.newaxis,:])
    #-- calculate Euler's of spherical harmonic order multiplied by azimuth phi
    mphi = np.exp(1j*np.dot(np.squeeze(phi)[:,np.newaxis],m[np.newaxis,:]))
    #-- calculate spherical harmonics
    Ylms = F*Pl[:,0:l+1]*mphi
    #-- multiply by C and convert to reduced matrix (theta,Slm:Clm)
    SPH = C*np.concatenate((np.imag(Ylms[:,:0:-1]),np.real(Ylms)), axis=1)
    return SPH

#-- RBF interpolant with shape parameter epsilon through the node points
#-- (theta,phi) with function values f from Fornberg
#-- Outputs beta: the expansion coefficients of the interpolant with respect to
#-- the RBF_QR basis.
def RBF_QR(theta,phi,epsilon,data,RBF):
    n = len(phi)
    Y1 = np.zeros((n,n))
    B1 = np.zeros((n,n))
    #-- counter variable for filling spherical harmonic matrix
    index = 0
    #-- difference adding the next spherical harmonic degree
    d = 0.0
    #-- degree of the n_th spherical harmonic
    l = 0
    l_n = np.ceil(np.sqrt(n))-1
    #-- floating point machine precision
    eps = np.finfo(np.float).eps
    while (d < -np.log10(eps)):
        #-- create new variables for Y and B which will resize if (l > (l_n -1))
        lmax = np.max([l_n,l])
        Y = np.zeros((n,int((lmax+1)**2)))
        Y[:,:index] = Y1[:,:index]
        B = np.zeros((n,int((lmax+1)**2)))
        B[:,:index] = B1[:,:index]
        #-- Each loop adds a block of columns of SPH of degree l to Y and to B.
        #-- Compute the spherical harmonics matrix
        Y[:,index:2*l+index+1] = spherical_harmonic_matrix(l,theta,phi)
        #-- Compute the expansion coefficients matrix
        B[:,index:2*l+index+1] = Y[:,index:2*l+index+1]*RBF(epsilon,l)
        B[:,index+l] = B[:,index+l]/2.0
        #-- Truncation criterion
        if (l > (l_n - 1)):
            dN1 = np.linalg.norm(B[:,int(l_n**2):int((l_n+1)**2)], ord=np.inf)
            dN2 = np.linalg.norm(B[:,int((l+1)**2)-1], ord=np.inf)
            d = np.log10(dN1/dN2*epsilon**(2*(l_n-l)))
        #-- copy B to B1 and Y to Y1
        B1 = B.copy()
        Y1 = Y.copy()
        #-- Calculate column index of next block
        index += 2*l+1
        l += 1
    #-- QR-factorization to find the RBF_QR basis
    _,R = np.linalg.qr(B)
    #-- Introduce the powers of epsilon
    X1 = np.kron(np.ones((n,1)), np.ceil(np.sqrt(np.arange(n,l**2))))
    X2 = np.kron(np.ones((1,l**2-n)),
        (np.ceil(np.sqrt(np.arange(1,n+1)))-1)[:,np.newaxis])
    E = epsilon**(2.0*(X1 - X2))
    #-- Solve the interpolation linear system
    R_beta = np.transpose(E*np.linalg.lstsq(R[:n,:n], R[:n,n:], rcond=-1)[0])
    R_new = np.concatenate((np.eye(n),R_beta),axis=0)
    w = np.linalg.lstsq(np.dot(Y,R_new), data, rcond=-1)[0]
    return (R_new,w)


def xyz2sph(x,y,z):
    """
    xyz2sph.py
    Written by Tyler Sutterley (UPDATED 08/2016)
    Converts Cartesian coordinates to spherical coordinates
    CALLING SEQUENCE:
        sph = xyz2sph(x,y,z)
        lon = sph['lon']
        lat = sph['lat']
        rad = sph['rad']
    INPUTS:
        x,y,z in cartesian coordinates
    OUTPUTS:
        lon: spherical longitude
        lat: spherical latitude
        rad: spherical radius
    """
    import numpy as np

    #-- calculate radius
    rad = np.sqrt(x**2.0 + y**2.0 + z**2.0)

    #-- calculate angular coordinates
    #-- phi: azimuthal angle
    phi = np.arctan2(y,x)
    #-- th: polar angle
    th = np.arccos(z/rad)

    #-- convert to degrees and fix to 0:360
    lon = 180.0*phi/np.pi
    ii = np.nonzero(lon < 0)
    count = np.count_nonzero(lon < 0)
    if (count != 0):
        lon[ii] = lon[ii]+360.0
    #-- convert to degrees and fix to -90:90
    lat = 90.0 -(180.0*th/np.pi)

    return {'lon': lon, 'lat':lat, 'rad':rad}

