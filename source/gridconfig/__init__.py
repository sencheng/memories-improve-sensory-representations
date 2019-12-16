"""
This package contains the settings for the simulation runs to start with :py:mod:`grid`.
The settings are defined by two types of files in the *gridconfig* folder:

1. **sfadef** files
2. :py:mod:`gridconfig.params` file

------------------
sfadef files
------------------

These files define the SFA parameters used for SFA1 and SFA2. For each of these files,
the entire grid of parameter combinations defined in :py:mod`gridconfig:params` is executed.

sfadef files have to be named as follows: **sfadef[index].py**
*index* can be any numbers and there can be any number of sfadef files.

Each sfadef file has to include two SFA parameter lists, one named *sfa1*
and the other named *sfa2*. How to define such a parameter list,
see :py:mod:`core.semantic_params`.

Example file ``sfadef1.py``::

    sfa1 = [
            ('layer.linear', {
                'bo': 30,
                'rec_field_ch': 18,
                'spacing': 6,
                'in_channel_dim': 1,
                'out_sfa_dim': 32
            })
        ]

    sfa2 = [
            ('inc.linear', {
                'dim_in': 288,
                'dim_out': 16
            })
        ]

"""

from os import listdir
import os
import re

from . import params

importlist = []

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

this_file = os.path.abspath(__file__)
this_path = this_file[:-len("/__init__.py")]
# print(this_path)

for file in listdir(this_path):  # required to get absolute path because relative path does not work with documentation AND calling from source
    if file.startswith("sfadef") and file[::-1].startswith("yp."):
        importlist.append(file.split(".")[0])
        exec("from . import " + importlist[-1])

importlist.sort(key=natural_keys)
