#!/bin/sh

DIR=/discover/nobackup/projects/gmao/gmao_ops/pub//f5295_fp/das/Y2024/M01
COLLECTIONS="inst3_3d_ext_Np inst3_2d_met_Nx inst1_2d_hwl_Nx inst3_3d_aer_Np inst3_3d_chm_Np inst3_3d_asm_Np tavg1_2d_slv_Nx"

for collection in $COLLECTIONS; do

  result=`find $DIR -type f -name "*${collection}*"`
  if [ -n "$result" ]; then
    echo $collection
  fi

done

exit 0
