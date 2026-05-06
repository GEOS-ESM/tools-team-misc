#! /usr/bin/env python

import os
import sys
import numpy as np
from netCDF4 import Dataset

vout = np.full((3,3,4), -9999.0, np.float32)
vout = np.ndarray((3,3,4), dtype=np.float32)
vout[:] = -9999.0

print vout
