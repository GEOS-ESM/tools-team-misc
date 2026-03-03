#!/bin/bash
#SBATCH --partition=geosgms
#SBATCH --constraint=mil
#SBATCH --nodes=1
#SBATCH --time=0:15:00
#SBATCH --job-name=WxMaps
##SBATCH --output=/dev/null

umask 022

source /usr/share/lmod/lmod/init/bash
module load python/GEOSpyD
module load ffmpeg

# 1. Create a job-specific local cache on the compute node
#export LOCAL_CACHE="$TMPDIR/pycache_tmp"
#mkdir -p "$LOCAL_CACHE"

# 2. Redirect Python to use this local directory instead of GPFS
#export PYTHONPYCACHEPREFIX="$LOCAL_CACHE"

# Optional: Ensure everything is clean for this run
# export PYTHONDONTWRITEBYTECODE=1 

#python plotall.py \
#    --data-reader gencast_geos_fp \
#    --exp-path "/discover/nobackup/projects/gmao/osse2/GenCast_FP" \
#    --exp-res 100KM \
#    --exp-id f5421_fpp \
#    --product sea_level_pressure \
#    --nproc 1 \
#    --fdate 20260129_00z \
#    --pdate 20260201_1200z \
#    --map-type conus \
#    --style light \
#    --boundaries coastlines countries states
#exit 0

#python plotall.py \
#    --data-reader geos_forward_processing \
#    --fp-base-path /discover/nobackup/projects/gmao/gmao_ops/pub \
#    --exp-res 12KM \
#    --exp-id f5295_fp \
#    --product sea_level_pressure \
#    --nproc 1 \
#    --fdate 20260129_00z \
#    --pdate 20260201_1200z \
#    --map-type conus \
#    --style light \
#    --boundaries coastlines countries states
#exit 0

python plotall.py \
    --product max_reflectivity \
    --nproc 1 \
    --fdate 20260202_00z \
    --pdate 20260202_1600 \
    --map-type conus \
    --style light
exit 0

python plotall.py --product max_reflectivity \
    --fdate 20260119_00z \
    --pdate 20260119_1200 \
    --map-type goes_east_full_disk \
    --style satellite
exit 0

product="max_reflectivity"
region="goes_east_full_disk"
style="satellite"
exp_res="CONUS02KM"
exp_id="Feature-c2160_L137"
fdate="20260119_00z"
nproc=1

python plotall.py --product $product \
                  --nproc $nproc \
                  --fdate $fdate \
                  --map-type $region \
                  --show-nws-warnings \
                  --style $style \
                  --boundaries coastlines countries states

./animate_frames_opt.sh $product $region $exp_res $exp_id $fdate

exit 0
python plotall.py --product basemap \
    --fdate 20260116_00z --pdate 20260116_1200z \
    --map-type conus \
    --style wxmaps \
    --boundaries coastlines countries states


exit 0
    --use-base-image --base-image-type bmng_no_snow \
    --base-image-interpolation none 
exit 0
    --show-nws-warnings
exit 0

python create_basemaps.py --map-type conus --fdate 20260116_00z --pdate 20260116_1200z \
    --show-nws-warnings \
    --ocean-color "#EEEEEE" --land-color "#FFFFFF" \
    --state-color "#999999" --state-width 0.5 --state-alpha 0.6
exit 0
    --use-gshhs --gshhs-resolution h \
    --roads --road-color "#A10000" --road-width 0.25 --road-alpha 0.4 --major-roads-only \
exit 0

# 2. Create multiple map types
python create_basemaps.py --map-type conus europe global \
    --fdate 20260116_00z --pdate 20260118_1200z

# 3. Create map with all boundary features
python create_basemaps.py --map-type conus \
    --fdate 20260116_00z --pdate 20260118_1200z \
    --boundaries coastlines countries states counties rivers \
    --roads --cities

# 4. Batch process multiple times
python batch_create_basemaps.py \
    --map-types conus conus_east conus_west \
    --fdate 20260116_00z \
    --start-pdate 20260116_00z \
    --end-pdate 20260118_00z \
    --interval 6 \
    --nprocs 8

# 5. List all available map types
python create_basemaps.py --list-maps
