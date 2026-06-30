#! /usr/bin/env python

import sys
import os
from PIL import Image, ImageOps

from imutils import imbright

Image.MAX_IMAGE_PIXELS = None

fname = sys.argv[1]
oname = sys.argv[2]

img = Image.open(fname).convert("LA")
img = imbright(img, 1.8)
img.save(oname, format='png')
