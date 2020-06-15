# 气象应用开发基本程序库
提供一些关于气象科学计算的基础功能函数，包括数组处理、数学函数、物理常数、时间处理、客观分析等模块;  
以及气象诊断分析程序，包括动力, 热力, 水汽和天气特征分析等。

Only Python 3 is supported.
建议安装[Anaconda](https://www.anaconda.com/products/individual)数据科学工具库,
已包括scipy, numpy, matplotlib等大多数常用科学程序库.

## Dependencies
Other required packages:

- [Numpy](https://numpy.org/), `conda install -c conda-forge numpy`
- [Scipy](http://www.scipy.org/), `conda install -c conda-forge scipy`
- [Pandas](http://pandas.pydata.org/), `conda install -c conda-forge pandas`
- [Xarray](https://github.com/pydata/xarray), `conda install -c conda-forge xarray`
- [Numba](http://numba.pydata.org/), `conda install -c numba numba`
- [Pyproj](https://github.com/pyproj4/pyproj), `conda install -c conda-forge pyproj`
- [Python-dateutil](https://pypi.org/project/python-dateutil/), `conda install -c conda-forge python-dateutil`
- [Metpy](https://github.com/Unidata/MetPy), `conda install -c conda-forge metpy`
- [Pyinterp](https://github.com/CNES/pangeo-pyinterp), `conda install -c conda-forge pyinterp`

## Install
Using the fellowing command to install packages:
```
  pip install git+git://github.com/nmcdev/nmc_met_base.git
```

or download the package and install:
```
  git clone --recursive https://github.com/nmcdev/nmc_met_base.git
  cd nmc_met_base
  python setup.py install
```
