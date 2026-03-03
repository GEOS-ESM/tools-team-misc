#!/bin/sh

#tr '\r' '\n' < $1 > temp
#mv temp $1

python xml2cmap.py $1

exit 0
