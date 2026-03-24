#!/bin/bash

#NOTE: Running using the uv python config

bindir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Test uv setup works
# script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# project_root="$(dirname "$script_dir")"
# cd "$project_root"
# uv run tests/test_uv.py
# 

PYTHONPATH="$srcdir${PYTHONPATH:+:$PYTHONPATH}" \
    python "$bindir/plotall.py" \
    --product temperature_2m_EarthNow \
    --nproc 1 \
    --fdate 20260202_00z \
    --pdate 20260202_1600 \
    --map-type global \
    --base-path /discover/nobackup/$USER/EarthNow/plots \
    --style light \
    --boundaries countries \
    --boundaries states 
exit 0
