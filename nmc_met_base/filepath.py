# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
  Manipulate the file name and systems.
"""


import os
import pathlib
import datetime
import numpy as np
import pandas as pd


def product_filename(model=None, product=None, level=None, obs_time=None,
                     init_time=None, fhour=None, valid_time=None,
                     statistic=None, place=None, suffix=None, root_dir=None):
    """
    Construct standard product file name, all parameters should not
    include "_".

    :param model: model name.
    :param product: product name.
    :param level: vertical level.
    :param obs_time: observation level.
    :param init_time: model initial model time.
    :param fhour: model forecast time.
    :param valid_time: model forecast valid time.
    :param statistic: statistic method name.
    :param place: place or station name.
    :param suffix: file suffix.
    :param root_dir: file directory.
    :return: product file name.
    """

    # define filename
    filename = ""

    # model name
    if model is not None:
        filename = filename + "MD_" + str(model).strip().upper()

    # product name
    if product is not None:
        filename = filename + "_PD_" + str(product).strip()

    # vertical level
    if level is not None:
        filename = filename + "_LV_" + str(level).strip()

    # observation time
    if obs_time is not None:
        if isinstance(obs_time, datetime.datetime):
            filename = filename + "_OT_" + obs_time.strftime("%Y%m%d%H")
        elif isinstance(obs_time, np.datetime64):
            filename = filename + "_OT_" + \
                       pd.to_datetime(str(obs_time)).strftime("%Y%m%d%H")
        else:
            filename = filename + "_OT_" + str(obs_time).strip()

    # model initial time
    if init_time is not None:
        if isinstance(init_time, datetime.datetime):
            filename = filename + "_IT_" + init_time.strftime("%Y%m%d%H")
        elif isinstance(init_time, np.datetime64):
            filename = filename + "_IT_" + \
                       pd.to_datetime(str(init_time)).strftime("%Y%m%d%H")
        else:
            filename = filename + "_IT_" + str(init_time).strip()

    # model forecast hour
    if fhour is not None:
        filename = filename + "_FH_" + str(fhour).strip()

    # model valid time
    if valid_time is not None:
        if isinstance(valid_time, datetime.datetime):
            filename = filename + "_VT_" + valid_time.strftime("%Y%m%d%H")
        elif isinstance(valid_time, np.datetime64):
            filename = filename + "_VT_" + \
                       pd.to_datetime(str(valid_time)).strftime("%Y%m%d%H")
        else:
            filename = filename + "_VT_" + str(valid_time).strip()

    # statistic name
    if statistic is not None:
        filename = filename + "_SN_" + str(statistic).strip()

    # place name
    if place is not None:
        filename = filename + "_PN_" + str(place).strip()

    # remove the first "_"
    if filename[0] == "_":
        filename = filename[1:]

    # add suffix
    if suffix is not None:
        if suffix[0] == ".":
            filename = filename + suffix
        else:
            filename = filename + "." + suffix

    # add root directory
    if root_dir is not None:
        filename = os.path.join(root_dir, filename)

    # return
    return filename


def product_filename_retrieve(filename):
    """
    Retrieve information from the standard product filename.

    :param filename: file name.
    :return: filename information dictionary.
    """

    file_name = pathlib.PureWindowsPath(filename)
    file_stem = file_name.stem.split("_")
    return dict(zip(file_stem[0::2], file_stem[1::2]))

