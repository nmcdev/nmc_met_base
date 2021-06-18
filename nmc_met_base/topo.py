# _*_ coding: utf-8 _*_

# Copyright (c) 2021 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
Compute DEM-based topographical descriptors, like the topographical position index (TPI) and the Sx,
The TPI describes the elevation of a given point relatively to its neighbourhood. The Sx is used to
describe the horizon in a given direction and spatial scale.

refer to https://github.com/MeteoSwiss/topo-descriptors
"""

import datetime as dt
import functools
import logging
import time
from pathlib import Path
from numba.np.ufunc import parallel

import numpy as np
import numpy.ma as ma
import xarray as xr
from numba import njit, prange
from scipy import ndimage, signal
from multiprocessing import Pool, Value, cpu_count

# Values lower or equal than min_elevation are filtered out
min_elevation: -100

# The number of standard deviations per unit scale
scale_std: 4

logger = logging.getLogger(__name__)


def get_dem_netcdf(path_dem):
    """Load the DEM into a xarray DataArray and filter NaNs
    Parameters
    ----------
    path_dem: string
        absolute or relative path to the DEM netCDF file.
    Returns
    -------
    xarray DataArray with the DEM values.
    """

    dem_ds = xr.open_dataset(path_dem, decode_times=False)
    dem_da = (
        dem_ds.to_array()
        .isel(variable=0, drop=True)
        .reset_coords(drop=True)
        .astype(np.float32)
    )

    return dem_da.where(dem_da > min_elevation)


def to_netcdf(array, coords, name, crop=None, outdir="."):
    """Save an array of topographic descriptors in NetCDF. It is first converted
    into a xarray DataArray with the same coordinates as the input DEM DataArray
    and a specified name.
    Parameters
    ----------
    array : array to be saved as netCDF
    coords : dict
        Coordinates for the array (i.e. those of the DEM).
    name : string
        Name for the array
    crop (optional) : dict
        The array is cropped to the given extend before being saved. Keys should
        be coordinates labels as in coords and values should be slices of [min,max]
        extend. Default is None.
    outdir (optional) : string
        The path to the output directory. Save to working directory by default.
    """

    name = str.upper(name)
    outdir = Path(outdir)
    da = xr.DataArray(array, coords=coords, name=name).sel(crop)
    filename = f"topo_{name}.nc"
    da.to_dataset().to_netcdf(outdir / filename)


def from_latlon(latitude, longitude, force_zone_number=None, force_zone_letter=None):
    """This function converts Latitude and Longitude to UTM coordinate
        Parameters
        ----------
        latitude: float or NumPy array
            Latitude between 80 deg S and 84 deg N, e.g. (-80.0 to 84.0)
        longitude: float or NumPy array
            Longitude between 180 deg W and 180 deg E, e.g. (-180.0 to 180.0).
        force_zone_number: int
            Zone number is represented by global map numbers of an UTM zone
            numbers map. You may force conversion to be included within one
            UTM zone number.  For more information see utmzones [1]_
        force_zone_letter: str
            You may force conversion to be included within one UTM zone
            letter.  For more information see utmzones [1]_
        Returns
        -------
        easting: float or NumPy array
            Easting value of UTM coordinates
        northing: float or NumPy array
            Northing value of UTM coordinates
        zone_number: int
            Zone number is represented by global map numbers of a UTM zone
            numbers map. More information see utmzones [1]_
        zone_letter: str
            Zone letter is represented by a string value. UTM zone designators
            can be accessed in [1]_
       .. _[1]: http://www.jaworski.ca/utmzones.htm
    """
    
    K0 = 0.9996

    E = 0.00669438
    E2 = E * E
    E3 = E2 * E
    E_P2 = E / (1 - E)

    SQRT_E = np.sqrt(1 - E)
    _E = (1 - SQRT_E) / (1 + SQRT_E)
    _E2 = _E * _E
    _E3 = _E2 * _E
    _E4 = _E3 * _E
    _E5 = _E4 * _E

    M1 = (1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256)
    M2 = (3 * E / 8 + 3 * E2 / 32 + 45 * E3 / 1024)
    M3 = (15 * E2 / 256 + 45 * E3 / 1024)
    M4 = (35 * E3 / 3072)

    R = 6378137

    ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"

    use_numpy = True

    class OutOfRangeError(ValueError):
        pass

    #--------------------------------------------------------------------------
    def in_bounds(x, lower, upper, upper_strict=False):
        if upper_strict and use_numpy:
                return lower <= np.min(x) and np.max(x) < upper
        elif upper_strict and not use_numpy:
            return lower <= x < upper
        elif use_numpy:
            return lower <= np.min(x) and np.max(x) <= upper
        return lower <= x <= upper

    #--------------------------------------------------------------------------
    def check_valid_zone(zone_number, zone_letter):
        if not 1 <= zone_number <= 60:
            raise OutOfRangeError('zone number out of range (must be between 1 and 60)')

        if zone_letter:
            zone_letter = zone_letter.upper()

            if not 'C' <= zone_letter <= 'X' or zone_letter in ['I', 'O']:
                raise OutOfRangeError('zone letter out of range (must be between C and X)')
    
    #--------------------------------------------------------------------------
    def mixed_signs(x):
        return use_numpy and np.min(x) < 0 and np.max(x) >= 0

    #--------------------------------------------------------------------------
    def negative(x):
        if use_numpy:
            return np.max(x) < 0
        return x < 0

    #--------------------------------------------------------------------------
    def mod_angle(value):
        """Returns angle in radians to be between -pi and pi"""
        return (value + np.pi) % (2 * np.pi) - np.pi

    #--------------------------------------------------------------------------
    def latitude_to_zone_letter(latitude):
        # If the input is a numpy array, just use the first element
        # User responsibility to make sure that all points are in one zone
        if use_numpy and isinstance(latitude, np.ndarray):
            latitude = latitude.flat[0]

        if -80 <= latitude <= 84:
            return ZONE_LETTERS[int(latitude + 80) >> 3]
        else:
            return None

    #--------------------------------------------------------------------------
    def latlon_to_zone_number(latitude, longitude):
        # If the input is a numpy array, just use the first element
        # User responsibility to make sure that all points are in one zone
        if use_numpy:
            if isinstance(latitude, np.ndarray):
                latitude = latitude.flat[0]
            if isinstance(longitude, np.ndarray):
                longitude = longitude.flat[0]

        if 56 <= latitude < 64 and 3 <= longitude < 12:
            return 32

        if 72 <= latitude <= 84 and longitude >= 0:
            if longitude < 9:
                return 31
            elif longitude < 21:
                return 33
            elif longitude < 33:
                return 35
            elif longitude < 42:
                return 37

        return int((longitude + 180) / 6) + 1

    #--------------------------------------------------------------------------
    def zone_number_to_central_longitude(zone_number):
        return (zone_number - 1) * 6 - 180 + 3

    #--------------------------------------------------------------------------
    if not in_bounds(latitude, -80, 84):
        raise OutOfRangeError('latitude out of range (must be between 80 deg S and 84 deg N)')
    if not in_bounds(longitude, -180, 180):
        raise OutOfRangeError('longitude out of range (must be between 180 deg W and 180 deg E)')
    if force_zone_number is not None:
        check_valid_zone(force_zone_number, force_zone_letter)

    lat_rad = np.radians(latitude)
    lat_sin = np.sin(lat_rad)
    lat_cos = np.cos(lat_rad)

    lat_tan = lat_sin / lat_cos
    lat_tan2 = lat_tan * lat_tan
    lat_tan4 = lat_tan2 * lat_tan2

    if force_zone_number is None:
        zone_number = latlon_to_zone_number(latitude, longitude)
    else:
        zone_number = force_zone_number

    if force_zone_letter is None:
        zone_letter = latitude_to_zone_letter(latitude)
    else:
        zone_letter = force_zone_letter

    lon_rad = np.radians(longitude)
    central_lon = zone_number_to_central_longitude(zone_number)
    central_lon_rad = np.radians(central_lon)

    n = R / np.sqrt(1 - E * lat_sin**2)
    c = E_P2 * lat_cos**2

    a = lat_cos * mod_angle(lon_rad - central_lon_rad)
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a

    m = R * (M1 * lat_rad -
             M2 * np.sin(2 * lat_rad) +
             M3 * np.sin(4 * lat_rad) -
             M4 * np.sin(6 * lat_rad))

    easting = K0 * n * (a +
                        a3 / 6 * (1 - lat_tan2 + c) +
                        a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * E_P2)) + 500000

    northing = K0 * (m + n * lat_tan * (a2 / 2 +
                                        a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c**2) +
                                        a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * E_P2)))

    if mixed_signs(latitude):
        raise ValueError("latitudes must all have the same sign")
    elif negative(latitude):
        northing += 10000000

    return easting, northing, zone_number, zone_letter


def scale_to_pixel(scales, dem_da):
    """Convert distances in meters to the closest odd number of pixels based on
    the DEM resolution.
    Parameters
    ----------
    scales : list of scalars
        Scales in meters on which we want to compute the topographic descriptor.
        Corresponds to the size of the squared kernel used to compute it.
    dem_da : xarray DataArray representing the DEM and its grid coordinates.
    Coordinates must be projected and named 'x', 'y'; or in WGS84 and named
    'lon', 'lat'. In the latter case, they are reprojected to UTM to derive the
    average resolution in meters.
    Returns
    -------
    list of int :
        Contain the corresponding scales in pixel size.
    dict with two 1-D or 2-D arrays :
        Resolution in meters of each DEM grid points in the x and y directions.
    """
    check_dem(dem_da)
    x_coords, y_coords = dem_da["x"].values, dem_da["y"].values
    epsg_code = dem_da.attrs["crs"].lower()
    if epsg_code == "epsg:4326":
        logger.warning(
            f"Reprojecting coordinates from WGS84 to UTM to obtain units of meters"
        )
        x_coords, y_coords = np.meshgrid(x_coords, y_coords)
        x_coords, y_coords, _, _ = from_latlon(y_coords, x_coords)
        x_coords, y_coords = x_coords.astype(np.float32), y_coords.astype(np.float32)

    n_dims = len(x_coords.shape)
    x_res = np.gradient(x_coords, axis=n_dims - 1)
    y_res = np.gradient(y_coords, axis=0)
    mean_res = np.mean(np.abs([x_res.mean(), y_res.mean()]))

    return round_up_to_odd(np.array(scales) / mean_res), {"x": x_res, "y": y_res}


def round_up_to_odd(f):
    """round float to the nearest odd integer"""

    return np.asarray(np.round((f - 1) / 2) * 2 + 1, dtype=np.int64)


def get_sigmas(smth_factors, scales_pxl):
    """Return scales expressed in standard deviations for gaussian filters.
    Parameters
    ----------
    smth_factors : list of scalars or None elements or a combination of both.
        Factors by which the scales in pixel must be multiplied. None or zeros
        results in None in the output.
    scales_pxl : list of int
        Scales expressed in pixels.
    Returns
    -------
    list of scalars/None elements representing scales in standard deviations.
    """

    sigmas = (
        [fact if fact else np.nan for fact in smth_factors] * scales_pxl / scale_std
    )

    return [None if np.isnan(sigma) else sigma for sigma in sigmas]


def fill_na(dem_da):
    """get indices of NaNs and interpolates them.
    Parameters
    ----------
    dem_da : xarray DataArray containing the elevation data.
    Returns
    -------
    ind_nans : tuple of two 1D arrays
        Contains the row / column indices of the NaNs in the original dem.
    Xarray DataArray with interpolated NaNs in x direction using "nearest" method.
    """

    ind_nans = np.where(np.isnan(dem_da))
    return ind_nans, dem_da.interpolate_na(
        dim=dem_da.dims[1], method="nearest", fill_value="extrapolate"
    )


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        t_start = time.monotonic()
        value = func(*args, **kwargs)
        t_elapsed = str(dt.timedelta(seconds=time.monotonic() - t_start)).split(".", 2)[
            0
        ]
        logger.info(f"Computed in {t_elapsed} (HH:mm:ss)")
        return value

    return wrapper_timer


def check_dem(dem):
    """
    Check that the input dem conforms to the data model, namely:
      - instance of xarray.DataArray
      - 2D field
      - y and x dimensions
      - crs attribute specifying an EPSG code.
    """
    if not isinstance(dem, xr.DataArray):
        raise ValueError("dem must be a xr.DataArray")
    if dem.ndim != 2:
        raise ValueError("dem must be a two-dimensional array")
    if dem.dims != ("y", "x"):
        raise ValueError("dem dimensions must be ('y', 'x')")
    if not "crs" in dem.attrs:
        raise KeyError("missing 'crs' (case sensitive) attribute in dem")
    if not "epsg:" in dem.attrs["crs"].lower():
        raise ValueError(
            "missing 'epsg:' (case insensitive) key in the 'crs' attribute"
        )


def compute_tpi(dem_da, scales, smth_factors=None, ind_nans=[], crop=None, outdir="."):
    """Wrapper to 'tpi' function to launch computations for all scales and save
    outputs as netCDF files.

    Parameters
    ----------
    dem_da : xarray DataArray representing the DEM and its grid coordinates.
    scales : scalar or list of scalars
        Scale(s) in meters on which we want to compute the TPI.
        Corresponds to the diameter of the kernel used to compute it.
    smth_factors (optional) : scalar or None or list with a combination of both.
        Fraction(s) of the scale(s) at which the DEM is smoothed first (with a
        gaussian filter). If None (default), no prior smoothing is performed.
        If a scalar, the same fraction is used to determine the smoothing scale
        of all specified scales. If a list, must match the length of arg 'scales'.
    ind_nans (optional) : tuple of two 1D arrays
        Contains the (row, column) indices of the NaNs in the original DEM to be
        reassigned after computations. NaNs in the original DEM should be
        interpolated prior computations as they propagate in convolutions with
        the fast Fourier transform method (scipy.signal.convolve).
    crop (optional) : dict
        If specified the outputs are cropped to the given extend. Keys should be
        the coordinates labels of dem_da and values should be slices of [min,max]
        extend. Default is None.
    outdir (optional) : string
        The path to the output directory. Save to working directory by default.

    See also
    --------
    tpi, circular_kernel
    """

    check_dem(dem_da)
    logger.info(f"***Starting TPI computation for scales {scales} meters***")
    if not hasattr(scales, "__iter__"):
        scales = [scales]
    if not hasattr(smth_factors, "__iter__"):
        smth_factors = [smth_factors] * len(scales)

    scales_pxl, _ = scale_to_pixel(scales, dem_da)
    sigmas = get_sigmas(smth_factors, scales_pxl)

    for idx, scale_pxl in enumerate(scales_pxl):
        logger.info(
            f"Computing scale {scales[idx]} meters with smoothing factor"
            f" {smth_factors[idx]} ..."
        )
        name = _tpi_name(scales[idx], smth_factors[idx])
        array = tpi(dem=dem_da.values, size=scale_pxl, sigma=sigmas[idx])

        array[ind_nans] = np.nan
        to_netcdf(array, dem_da.coords, name, crop, outdir)
        del array


@timer
def tpi(dem, size, sigma=None):
    """Compute the TPI over a digital elevation model. The TPI represents
    the elevation difference of a pixel relative to its neighbors.

    Parameters
    ----------
    dem : array representing the DEM.
    size : int
        Size of the kernel for the convolution. Represents the diameter (i.e. scale)
        in pixels at which the TPI is computed.
    sigma (optional) : scalar
        If provided, the DEM is first smoothed with a gaussian filter of standard
        deviation sigma (in pixel size).

    Returns
    -------
    array with TPI values

    See also
    --------
    scipy.signal.convolve, scipy.ndimage.gaussian_filter
    """

    kernel = circular_kernel(size)
    # exclude mid point from the kernel
    kernel[int(size / 2), int(size / 2)] = 0

    if sigma:
        dem = ndimage.gaussian_filter(dem, sigma)

    conv = signal.convolve(
        dem, kernel, mode="same"
    )  # ndimage.convolve(dem, kernel, mode='reflect')
    return dem - conv / np.sum(kernel)


def _tpi_name(scale, smth_factor):
    """Return name for the array in output of the tpi function"""

    add = f"_SMTHFACT{smth_factor:.3g}" if smth_factor else ""
    return f"TPI_{scale}M{add}"


def circular_kernel(size):
    """Generate a circular kernel.

    Parameters
    ----------
    size : int
        Size of the circular kernel (its diameter). For size < 5, the kernel is
        a square instead of a circle.

    Returns
    -------
    2-D array with kernel values
    """

    middle = int(size / 2)
    if size < 5:
        kernel = np.ones((size, size), dtype=np.float32)
    else:
        xx, yy = np.mgrid[:size, :size]
        circle = (xx - middle) ** 2 + (yy - middle) ** 2
        kernel = np.asarray(circle <= (middle ** 2), dtype=np.float32)

    return kernel


def compute_std(dem_da, scales, smth_factors=None, ind_nans=[], crop=None, outdir="."):
    """Wrapper to 'std' function to launch computations for all scales and save
    outputs as netCDF files.

    Parameters
    ----------
    dem_da : xarray DataArray representing the DEM and its grid coordinates.
    scales : scalar or list of scalars
        Scale(s) in meters on which we want to compute the TPI.
        Corresponds to the diameter of the kernel used to compute it.
    smth_factors (optional) : scalar or None or list with a combination of both.
        Fraction(s) of the scale(s) at which the DEM is smoothed first (with a
        gaussian filter). If None (default), no prior smoothing is performed.
        If a scalar, the same fraction is used to determine the smoothing scale
        of all specified scales. If a list, must match the length of arg 'scales'.
    ind_nans (optional) : tuple of two 1D arrays
        Contains the (row, column) indices of the NaNs in the original DEM to be
        reassigned after computations. NaNs in the original DEM should be
        interpolated prior computations as they propagate in convolutions with
        the fast Fourier transform method (scipy.signal.convolve).
    crop (optional) : dict
        If specified the outputs are cropped to the given extend. Keys should be
        the coordinates labels of dem_da and values should be slices of [min,max]
        extend. Default is None.
    outdir (optional) : string
        The path to the output directory. Save to working directory by default.

    See also
    --------
    std, circular_kernel
    """

    check_dem(dem_da)
    logger.info(f"***Starting STD computation for scales {scales} meters***")
    if not hasattr(scales, "__iter__"):
        scales = [scales]
    if not hasattr(smth_factors, "__iter__"):
        smth_factors = [smth_factors] * len(scales)

    scales_pxl, _ = scale_to_pixel(scales, dem_da)
    sigmas = get_sigmas(smth_factors, scales_pxl)

    for idx, scale_pxl in enumerate(scales_pxl):
        logger.info(
            f"Computing scale {scales[idx]} meters with smoothing factor"
            f" {smth_factors[idx]} ..."
        )
        name = _std_name(scales[idx], smth_factors[idx])
        array = std(dem=dem_da.values, size=scale_pxl, sigma=sigmas[idx])

        array[ind_nans] = np.nan
        to_netcdf(array, dem_da.coords, name, crop, outdir)
        del array


@timer
def std(dem, size, sigma=None):
    """Compute the standard deviation over a digital elevation model within
    a rolling window.

    Parameters
    ----------
    dem : array representing the DEM.
    size : int
        Size of the kernel for the convolution. Represents the diameter (i.e. scale)
        in pixels at which the std is computed.
    sigma (optional) : scalar
        If provided, the DEM is first smoothed with a gaussian filter of standard
        deviation sigma (in pixel size).

    Returns
    -------
    array with local standard deviation values

    See also
    --------
    scipy.signal.convolve, scipy.ndimage.gaussian_filter
    """
    kernel = circular_kernel(size)
    kernel_sum = np.sum(kernel)
    if sigma:
        dem = ndimage.gaussian_filter(dem, sigma)

    squared_dem = dem.astype("int32") ** 2
    sum_dem = signal.convolve(dem, kernel, mode="same")
    sum_squared_dem = signal.convolve(squared_dem, kernel, mode="same")

    variance = (sum_squared_dem - sum_dem ** 2 / kernel_sum) / (kernel_sum - 1)
    variance = np.clip(variance, 0, None)  # avoid small negative values

    return np.sqrt(variance)


def _std_name(scale, smth_factor):
    """Return name for the array in output of the tpi function"""

    add = f"_SMTHFACT{smth_factor:.3g}" if smth_factor else ""
    return f"STD_{scale}M{add}"


def compute_valley_ridge(
    dem_da,
    scales,
    mode,
    flat_list=[0, 0.15, 0.3],
    smth_factors=None,
    ind_nans=[],
    crop=None,
    outdir=".",
):
    """Wrapper to 'valley_ridge' function to launch computations for all scales
    and save outputs as netCDF files.

    Parameters
    ----------
    dem_da : xarray DataArray representing the DEM and its grid coordinates.
    scales : scalar or list of scalars
        Scale(s) in meters over which we want to compute the valley or the ridge
        index. Corresponds to the size of the squared kernel used to compute it.
    mode : {valley, ridge}
        Whether to compute the valley or ridge index.
    flat_list (optional) : list of floats in [0,1[
        Fractions of flat along the center line of the V-shape kernels. A certain
        amount of flat is use to approximate the shape of glacial valleys.
        Default is [0, 0.15, 0.3].
    smth_factors (optional) : scalar or None or list with a combination of both.
        Fraction(s) of the scale(s) at which the DEM is smoothed first (with a
        gaussian filter). If None (default), no prior smoothing is performed.
        If a scalar, the same fraction is used to determine the smoothing scale
        of all specified scales. If a list, must match the length of arg 'scales'.
    ind_nans (optional) : tuple of two 1D arrays
        Contains the (row, column) indices of the NaNs in the original DEM to be
        reassigned after computations. NaNs in the original DEM should be
        interpolated prior computations as they propagate in convolutions with
        the fast Fourier transform method (scipy.signal.convolve).
    crop (optional) : dict
        If specified the outputs are cropped to the given extend. Keys should be
        the coordinates labels of dem_da and values should be slices of [min,max]
        extend. Default is None.
    outdir (optional) : string
        The path to the output directory. Save to working directory by default.

    See also
    --------
    valley_ridge, _valley_kernels, _valley_ridge_names
    """

    check_dem(dem_da)
    logger.info(f"***Starting {mode} index computation for scales {scales} meters***")
    if not hasattr(scales, "__iter__"):
        scales = [scales]
    if not hasattr(smth_factors, "__iter__"):
        smth_factors = [smth_factors] * len(scales)

    scales_pxl, _ = scale_to_pixel(scales, dem_da)
    sigmas = get_sigmas(smth_factors, scales_pxl)

    pool = Pool(processes=min(len(scales_pxl), cpu_count()))
    for idx, scale_pxl in enumerate(scales_pxl):
        logger.info(
            f"Computing scale {scales[idx]} meters with smoothing factor"
            f" {smth_factors[idx]} ..."
        )
        names = _valley_ridge_names(scales[idx], mode, smth_factors[idx])
        pool.apply_async(
            _valley_ridge_wrap,
            args=(
                dem_da,
                scale_pxl,
                mode,
                flat_list,
                sigmas[idx],
                names,
                ind_nans,
                crop,
                outdir,
            ),
        )

    pool.close()
    pool.join()


def _valley_ridge_wrap(
    dem_da, size, mode, flat_list, sigma, names, ind_nans, crop, outdir
):
    """Wrapper to valley_ridge and to_netcdf functions to ease the parallelization
    of the different scales"""

    arrays = valley_ridge(dem_da.values, size, mode, flat_list, sigma)
    for array, name in zip(arrays, names):
        array[ind_nans] = np.nan
        to_netcdf(array, dem_da.coords, name, crop, outdir)


@timer
def valley_ridge(dem, size, mode, flat_list=[0, 0.15, 0.3], sigma=None):
    """Compute the valley or ridge index over a digital elevation model.
    The valley/ridge index highlights valley or ridges  at various scales.

    Parameters
    ----------
    dem : array representing the DEM.
    size : int
        Size of the kernel for the convolution. Represents the width (i.e. scale)
        in pixels of the valleys we are trying to highlight.
    mode : {valley, ridge}
        Whether to compute the valley or ridge index.
    flat_list (optional) : list of floats in [0,1[
        Fractions of flat along the center line of the V-shape kernels. A certain
        amount of flat is use to approximate the shape of glacial valleys.
        Default is [0, 0.15, 0.3].
    sigma (optional) : scalar
        If provided, the DEM is first smoothed with a gaussian filter of standard
        deviation sigma (in pixel size).

    Returns
    -------
    list of two arrays :
        First element is the norm and second the direction of the valley or ridge
        index. The direction in degrees is defined from 0° to 179°, increasing
        clockwise. W-E oriented valleys have a direction close to 0° or 180°,
        while S-N oriented valleys have a direction close to 90°.

    See also
    --------
    scipy.signal.convolve, scipy.ndimage.gaussian_filter
    """

    if mode not in ("valley", "ridge"):
        raise ValueError(f"Unknown mode {mode!r}")

    if sigma:
        dem = ndimage.gaussian_filter(dem, sigma)

    dem = (dem - dem.mean()) / dem.std()
    n_y, n_x = dem.shape
    dem = np.broadcast_to(dem, (len(flat_list), n_y, n_x))
    angles = np.arange(0, 180, dtype=np.float32)
    index_norm = np.zeros((n_y, n_x), dtype=np.float32) - np.inf
    index_dir = np.empty((n_y, n_x), dtype=np.float32)

    if mode == "ridge":
        kernels = _ridge_kernels(size, flat_list)
    else:
        kernels = _valley_kernels(size, flat_list)

    for angle in angles:  # 0° = E-W valleys, 90° = S-N valleys

        kernels_rot = _rotate_kernels(kernels, angle)
        conv = signal.convolve(dem, kernels_rot, mode="same")
        conv = np.max(conv, axis=0)
        bool_greater = conv > index_norm
        index_norm[bool_greater] = conv[bool_greater]
        index_dir[bool_greater] = angle
        del bool_greater
        if angle % 20 == 0:
            logger.info(f"angle {int(angle)}/180 finished")

    index_norm = np.ndarray.clip(index_norm, min=0)
    return [index_norm, index_dir]


def _valley_ridge_names(scale, mode, smth_factor):
    """Return names for the arrays in output of the valley_ridge function"""

    add = f"_SMTHFACT{smth_factor:.3g}" if smth_factor else ""
    name_norm = f"{mode}_NORM_{scale}M{add}"
    name_dir = f"{mode}_DIR_{scale}M{add}"

    return [name_norm, name_dir]


def _valley_kernels(size, flat_list):
    """Generate normalized V-shape and U-shape kernels to compute valley index.

    Parameters
    ----------
    size : int
        Size of the kernel.
    flat_list : list of floats in [0,1[
        Fractions of flat along the center line of the V-shape kernels. A certain
        amount of flat is use to approximate the shape of glacial valleys.

    Returns
    -------
    3-D array with 2-D kernels for each specified flat fraction.
    """

    middle = int(np.floor(size / 2))
    kernel_tmp = np.broadcast_to(np.arange(0, middle + 1), (size, middle + 1)).T
    kernel_tmp = np.concatenate(
        (np.flip(kernel_tmp[1:, :], axis=0), kernel_tmp), axis=0
    )
    kernel_tmp = np.asarray(kernel_tmp, dtype=np.float32)
    kernels = np.broadcast_to(kernel_tmp, (len(flat_list), size, size)).copy()

    for ind, flat in enumerate(flat_list):
        halfwidth = int(np.floor(np.floor(size * flat / 2) + 0.5))
        kernels[ind, middle - halfwidth : middle + halfwidth + 1, :] = kernels[
            ind, middle - halfwidth, 0
        ]
        kernels = (kernels - np.mean(kernels, axis=(1, 2), keepdims=True)) / np.std(
            kernels, axis=(1, 2), keepdims=True
        )

    return kernels


def _ridge_kernels(size, flat_list):
    """Generate normalized flipped V-shape and U-shape kernels to compute ridge index.

    Parameters
    ----------
    size : int
        Size of the kernel.
    flat_list : list of floats in [0,1[
        Fractions of flat along the center line of the V-shape kernels. A certain
        amount of flat is use to approximate the shape of glacial valleys.

    Returns
    -------
    3-D array with 2-D kernels for each specified flat fraction.
    """

    return _valley_kernels(size, flat_list) * -1


def _rotate_kernels(kernel, angle):
    """Rotate a 3-D kernel in the plane given by the last two axes"""

    kernels_rot = ndimage.rotate(
        kernel, angle, axes=(1, 2), reshape=True, order=2, mode="constant", cval=-9999
    )
    kernels_rot = ma.masked_array(kernels_rot, mask=kernels_rot == -9999)
    kernels_rot = (
        kernels_rot - np.mean(kernels_rot, axis=(1, 2), keepdims=True)
    ) / np.std(kernels_rot, axis=(1, 2), keepdims=True)
    return ma.MaskedArray.filled(kernels_rot, 0).astype(np.float32)


def compute_gradient(dem_da, scales, sig_ratios=1, ind_nans=[], crop=None, outdir="."):
    """Wrapper to 'gradient' function to launch computations for all scales
    and save outputs as netCDF files.

    Parameters
    ----------
    dem_da : xarray DataArray representing the DEM and its grid coordinates.
    scales : scalar or list of scalars
        Scale(s) in meters on which we want to compute the the valley or ridge
        index. Corresponds to the size of the squared kernel used to compute it.
    sig_ratios (optional) : scalar or list of scalars.
        Ratios w.r.t scales to define the smoothing scale in the perpendicular
        direction of the directional derivatives. If a list, must match the length
        of arg 'scales'. Default is 1 (i.e. same smoothing on both directions for
        all scales)
    ind_nans (optional) : tuple of two 1D arrays
        Contains the (row, column) indices of the NaNs in the original DEM to be
        reassigned after computations. NaNs in the original DEM should be
        interpolated prior computations as they propagate in convolutions.
    crop (optional) : dict
        If specified the outputs are cropped to the given extend. Keys should be
        the coordinates labels of dem_da and values should be slices of [min,max]
        extend. Default is None.
    outdir (optional) : string
        The path to the output directory. Save to working directory by default.

    See also
    --------
    gradient, sobel, _gradient_names
    """

    check_dem(dem_da)
    logger.info(f"***Starting gradients computation for scales {scales} meters***")
    if not hasattr(scales, "__iter__"):
        scales = [scales]
    if not hasattr(sig_ratios, "__iter__"):
        sig_ratios = [sig_ratios] * len(scales)

    scales_pxl, res_meters = scale_to_pixel(scales, dem_da)
    sigmas = scales_pxl / scale_std

    for idx, sigma in enumerate(sigmas):
        logger.info(
            f"Computing scale {scales[idx]} meters with sigma ratio "
            f"{sig_ratios[idx]} ..."
        )
        names = _gradient_names(scales[idx], sig_ratios[idx])
        arrays = gradient(
            dem=dem_da.values,
            sigma=sigma,
            res_meters=res_meters,
            sig_ratio=sig_ratios[idx],
        )

        for array, name in zip(arrays, names):
            array[ind_nans] = np.nan
            to_netcdf(array, dem_da.coords, name, crop, outdir)

        del arrays


@timer
def gradient(dem, sigma, res_meters, sig_ratio=1):
    """Compute directional derivatives, slope and aspect over a digital elevation
    model.

    Parameters
    ----------
    dem : array representing the DEM.
    sigma : scalar
        Standard deviation for the gaussian filters. This is set at 1/4 of the
        scales to which topo descriptors are computed (i.e. scale = 4*sigma)
    res_meters : dict with two 1-D or 2-D arrays
        Resolution in meters of each DEM grid points in the x and y directions.
        This is the second element returned by the function scale_to_pixel.
    sig_ratio (optional) : scalar
        Ratio w.r.t sigma to define the standard deviation of the gaussian in
        the direction perpendicular to the derivative. Default is 1 (i.e. same
        smoothing on both directions).

    Returns
    -------
    list of four arrays :
        First element is the W-E derivative, second the S-N derivative, third
        the slope (i.e. magnitude of the gradient) and fourth the aspect (i.e.
        direction of the gradient).

    See also
    --------
    scipy.ndimage.gaussian_filter, np.gradient, topo_helpers.scale_to_pixel
    """

    if sigma <= 1:
        dx, dy = sobel(dem)  # for lowest scale, use sobel filter instead
    elif sig_ratio == 1:  # faster solution when sig_ratio is 1
        dy, dx = np.gradient(ndimage.gaussian_filter(dem, sigma))
    else:
        sigma_perp = sigma * sig_ratio
        dx = np.gradient(ndimage.gaussian_filter(dem, (sigma_perp, sigma)), axis=1)
        dy = np.gradient(ndimage.gaussian_filter(dem, (sigma, sigma_perp)), axis=0)

    _normalize_dxy(dx, dy, res_meters)

    slope = np.sqrt(dx ** 2, dy ** 2)
    aspect = (
        180 + np.degrees(np.arctan2(dx, dy))
    ) % 360  # north faces = 0°, east faces = 90°

    return [dx, dy, slope, aspect]


def _gradient_names(scale, sig_ratio):
    """Return names for the arrays in output of the gradient function"""

    name_dx = f"WE_DERIVATIVE_{scale}M_SIGRATIO{sig_ratio:.3g}"
    name_dy = f"SN_DERIVATIVE_{scale}M_SIGRATIO{sig_ratio:.3g}"
    name_slope = f"SLOPE_{scale}M_SIGRATIO{sig_ratio:.3g}"
    name_aspect = f"ASPECT_{scale}M_SIGRATIO{sig_ratio:.3g}"

    return [name_dx, name_dy, name_slope, name_aspect]


def sobel(dem):
    """Compute directional derivatives, based on the sobel filter. The sobel is
    a combination of a derivative and a smoothing filter. It is defined over a
    3x3 kernel.

    Parameters
    ----------
    dem : array representing the DEM.

    Returns
    -------
    dx : 2-D array
        Derivative in the x direction (i.e. along second axis)
    dy : 2-D array
        Derivative in the y direction (i.e. along first axis)

    See also
    --------
    ndimage.convolve
    """

    sobel_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)

    sobel_kernel /= np.sum(np.abs(sobel_kernel))
    dx = ndimage.convolve(dem, sobel_kernel)
    dy = ndimage.convolve(dem, sobel_kernel.T)

    return dx, dy


def _normalize_dxy(dx, dy, res_meters):
    """Normalize directional derivatives, based on (projected) grid resolution.
    This is useful when the DEM does not come on an equidistant projected grid.
    Normalization occurs 'in place'.

    Parameters
    ----------
    dx : 2-D array
        Derivative in the x direction (i.e. along second axis)
    dy : 2-D array
        Derivative in the y direction (i.e. along first axis)
    res_meters : dict with two 1-D or 2-D arrays
        Resolution in meters of each DEM grid points in the x and y directions.
        This is the second element returned by the function scale_to_pixel.

    See also
    --------
    topo_helpers.scale_to_pixel
    """

    mean_res = np.mean(np.abs([res_meters["x"].mean(), res_meters["y"].mean()]))
    x_res = res_meters["x"] / mean_res
    y_res = res_meters["y"] / mean_res
    if len(y_res.shape) == 1:
        y_res = y_res[:, np.newaxis]

    dx /= x_res
    dy /= y_res


def compute_sx(
    dem_da,
    azimuth,
    radius,
    height=10.0,
    azimuth_arc=10.0,
    azimuth_steps=15,
    radius_min=0.0,
    crop=None,
    outdir=".",
):
    """Wrapper to 'Sx' function to launch computations and save
    outputs as netCDF files.

    Parameters
    ----------
    dem_da : xarray DataArray representing the DEM and its grid coordinates.
    azimuth : scalar
        Azimuth angle in degrees for imaginary lines.
    radius : scalar
        Maximum distance in meters for the imaginary lines.
    azimuth_arc (optional): scalar
        Angle of the circular sector centered around 'azimuth'.
    azimuth_steps (optional):
        Number of lines traced to find pixels within the circular sector.
        A higher number leads to more precise but longer computations.
    radius_min (optional): scalar
        Minimum value of radius below which pixels are excluded from imaginary lines.
    height (optional): scalar
        Parameter that accounts for instrument heights and
        reduce the impact of small proximal terrain perturbations.
    crop (optional) : dict
        If specified the outputs are cropped to the given extend. Keys should be
        the coordinates labels of dem_da and values should be slices of [min,max]
        extend. Default is None.

    See also
    --------
    sx, _sx_distance, _sx_source_idx_delta, _sx_bresenhamlines, _sx_rolling
    """
    check_dem(dem_da)
    logger.info(
        f"***Starting Sx computation for azimuth {azimuth} meters and radius {radius}***"
    )

    array = sx(
        dem_da,
        azimuth,
        radius,
        height=height,
        azimuth_arc=azimuth_arc,
        azimuth_steps=azimuth_steps,
        radius_min=radius_min,
    )

    name = _sx_name(radius, azimuth)
    name = str.upper(name)
    to_netcdf(array, dem_da.coords, name, crop, outdir)


@timer
def sx(
    dem_da,
    azimuth,
    radius,
    height=10.0,
    azimuth_arc=10.0,
    azimuth_steps=15,
    radius_min=0.0,
):
    """Compute the Sx over a digital elevation model.

    The Sx represents the maximum slope among all imaginary lines connecting a
    given pixel with all the ones lying in a specific direction and within a
    specified distance (Winstral et al., 2017). The Sx is a proven wind-specific
    terrain parameterization, as it is able to differentiate the slopes based
    on given wind directions and identify sheltered and exposed locations with
    respect to the incoming wind.
    Note that this routine computes one azimuth at a time. It is accelerated with
    Numba's Just In Time compilation, but is still computationally expensive.

    Parameters
    ----------
    dem_da : xarray DataArray representing the DEM and its grid coordinates.
    azimuth : scalar
        Azimuth angle in degrees for imaginary lines.
    radius : scalar
        Maximum distance in meters for the imaginary lines.
    azimuth_arc (optional): scalar
        Angle of the circular sector centered around 'azimuth'.
        Set to zero in order to draw a single line.
    azimuth_steps (optional): scalar integer
        Number of lines traced to find pixels within the circular sector.
        A higher number leads to more precise but longer computations.
        Defaults to 1 when 'azimuth_arc' is 0.
    radius_min (optional): scalar
        Minimum value of radius below which pixels are excluded from imaginary lines.
    height (optional): scalar
        Parameter that accounts for instrument heights and
        reduce the impact of small proximal terrain perturbations.

    Returns
    -------
    array with Sx values for one azimuth

    See also
    --------
    _sx_distance, _sx_source_idx_delta, _sx_bresenhamlines, _sx_rolling
    """

    if not isinstance(dem_da, xr.DataArray):
        raise TypeError("Argument 'dem_da' must be a xr.DataArray.")

    if azimuth_arc == 0:
        azimuth_steps = 1

    # define all azimuths
    azimuths = np.linspace(
        azimuth - azimuth_arc / 2, azimuth + azimuth_arc / 2, azimuth_steps
    )

    # grid resolutions
    _, res_meters = scale_to_pixel(radius, dem_da)
    dx = res_meters["x"].mean()
    dy = res_meters["y"].mean()

    # horizontal distance in meters from center in a window of size 2*radius
    window_distance = _sx_distance(radius, dx, dy)

    # exclude pixels closer than radius_min
    window_distance[window_distance < radius_min] = np.nan

    # indices of pixels that lie at distance radius in direction azimuth
    window_center = np.floor(np.array(window_distance.shape) / 2)
    source_delta = _sx_source_idx_delta(azimuths, radius, dx, dy)
    source = (window_center + source_delta).astype(np.int)

    # indices of all pixels between source pixels and target (center)
    lines_indices = _sx_bresenhamlines(source, window_center)

    # compute Sx
    sx = _sx_rolling(dem_da.values, window_distance, lines_indices, height)

    return sx


def _sx_distance(radius, dx, dy):
    """Compute distance from center in meters in a window of size 'radius'."""

    dx_abs = np.abs(dx)
    dy_abs = np.abs(dy)
    radius_pxl = max(radius / dy_abs, radius / dx_abs)

    # initialize window
    window = 2 * radius_pxl + 1  # must be odd
    center = np.floor(window / 2)
    x = np.arange(window)
    y = np.arange(window)
    x, y = np.meshgrid(x, y)

    # calculate distances from center for all points in the window
    distances = np.sqrt((((y - center) * dy) ** 2) + ((x - center) * dx) ** 2)

    return distances


def _sx_source_idx_delta(azimuths, radius, dx, dy):
    """Compute indices of pixels that lie at a distance 'radius' from
    the target, in the direction of 'azimuths'.
    """

    azimuths_rad = np.deg2rad(azimuths)
    delta_y_idx = np.rint(radius / dy * np.cos(azimuths_rad))
    delta_x_idx = np.rint(radius / dx * np.sin(azimuths_rad))

    delta = np.column_stack([delta_y_idx, delta_x_idx])

    return delta.astype(np.int64)


def _sx_bresenhamlines(start, end):
    """Compute indices of all pixels that lie between two sets of pixels."""

    max_iter = np.max(np.max(np.abs(end - start), axis=1))
    npts, dim = start.shape

    slope = end - start
    scale = np.max(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    blines = start[:, np.newaxis, :] + normalizedslope[:, np.newaxis, :] * stepmat

    # Approximate to nearest int
    blines = np.array(np.rint(blines), dtype=start.dtype)

    # Stop lines before center
    bsum = np.abs(blines - end).sum(axis=2)
    mask = np.diff(bsum, prepend=bsum[:, 0:1]) <= 0
    blines = blines[mask].reshape(-1, start.shape[-1])
    mask = np.all(blines == end, axis=1)
    blines = blines[~mask]

    return blines


@njit(parallel=True)
def _sx_rolling(dem, distance, blines, height):
    """Compute Sx values for the array with a loop over all elements."""

    window = int(distance.shape[0] / 2)
    ny, nx = dem.shape
    distance = np.array(
        [distance[j, i] for j, i in list(zip(blines[:, 0], blines[:, 1]))]
    )
    blines_centered = blines - window

    sx = np.zeros_like(dem)
    for j in prange(window, ny - window):
        for i in prange(window, nx - window):

            j_blines = j + blines_centered[:, 0]
            i_blines = i + blines_centered[:, 1]
            dem_blines = np.array([dem[j, i] for j, i in list(zip(j_blines, i_blines))])

            # compute tangent z / distance between P0 and all points
            z = dem_blines - (dem[j, i] + height)
            elev_angle = np.rad2deg(np.arctan(z / distance))

            # find the maximum angle in the cone
            sx[j, i] = np.nanmax(elev_angle)

    return sx


def _sx_name(radius, azimuth):
    """Return name for the array in output of the Sx function"""

    add = f"_RADIUS{int(radius[0])}-{int(radius)}_AZIMUTH{int(azimuth)}"
    return f"SX{add}"


# TODO: Relief