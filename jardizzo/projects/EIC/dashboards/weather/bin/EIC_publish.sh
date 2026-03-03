#!/bin/sh

# Set up environment
# ==================

  dattim=$1
  config=$2

  EIC_pub_dir=`get_param.py $dattim $config EIC_pub_dir`
  EIC_data_dir=`get_param.py $dattim $config EIC_data_dir`

  mkdir -p $EIC_pub_dir/EIC

  /bin/rm -f $EIC_pub_dir/EIC/*.mp4
  /bin/cp -p $EIC_data_dir/mp4/*.mp4 $EIC_pub_dir/EIC

exit 0
