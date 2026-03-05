#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: create_mrf <png-file-to-convert>" 
  return 1
fi

filename_png="$1"
input="${filename_png}"
output="${filename_png::-3}mrf"

gdal_translate -of MRF $input $output

