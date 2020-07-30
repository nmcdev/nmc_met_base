# 气象应用开发基本程序库

提供一些关于气象科学计算的基础功能函数，包括数组处理、数学函数、物理常数、时间处理、客观分析等模块;  
以及气象诊断分析程序，包括动力, 热力, 水汽和天气特征分析等。

Only Python 3 is supported.
建议安装[Anaconda](https://www.anaconda.com/products/individual)数据科学工具库,
已包括scipy, numpy, matplotlib等大多数常用科学程序库.

## Install

Using the fellowing command to install packages:

* 使用pypi安装源安装(https://pypi.org/project/nmc-met-base/)
```
  pip install nmc-met-base
```
* 若要安装Github上的开发版(请先安装[Git软件](https://git-scm.com/)):
```
  pip install git+git://github.com/nmcdev/nmc_met_base.git
```
* 或者下载软件包进行安装:
```
  git clone --recursive https://github.com/nmcdev/nmc_met_base.git
  cd nmc_met_base
  python setup.py install
```

### 可选支持库:
* [pyinterp](https://github.com/CNES/pangeo-pyinterp), `conda install pyinterp -c conda-forge`
