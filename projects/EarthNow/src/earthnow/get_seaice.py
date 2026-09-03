#!/usr/bin/env python3

# Put this as a function in the existing scripts?
import os
import numpy as np
from scipy.io import FortranFile
from datetime import datetime


def get_seaice_map(year, month, day, hour):
    filename = f"/discover/nobackup/projects/gmao/share/dao_ops/fvInput/g5gcm/bcs/realtime/OSTIA_REYNOLDS/2880x1440/dataoceanfile_OSTIA_REYNOLDS_ICE.2880x1440.{year:04d}.data"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    target_date = datetime(year, month, day)

    with FortranFile(filename, "r") as f:
        # Read the first header (14 floats)
        # scipy.io.FortranFile handles the 4-byte F77 record markers automatically
        hdr = f.read_reals(dtype=np.float32)

        file_y = int(hdr[0])
        file_m = int(hdr[1])
        file_d = int(hdr[2])
        nx = int(hdr[12])
        ny = int(hdr[13])

        file_date = datetime(file_y, file_m, file_d)

        # Read the first data record
        # Note: We use order='F' (Fortran/Column-major order) to ensure the
        # array shape matches IDL's FLTARR(nx, ny) identically
        sice = f.read_reals(dtype=np.float32).reshape((nx, ny))

        # Calculate the byte size of one full day of data
        # IDL calculated this as: 4 + 14 (HDR floats) + nx*ny (SICE floats)
        # Why the extra 4? 2 records per day * two 4-byte markers per record = 16 bytes.
        # 16 bytes is exactly 4 float32s.
        byte_size = 16 + (14 * 4) + (nx * ny * 4)

        # Calculate days to skip to reach the day *before* the target date
        skip_days = (target_date - file_date).days - 1

        if skip_days > 0:
            # f._fp accesses the underlying python file object to perform an absolute skip
            f._fp.seek(skip_days * byte_size, os.SEEK_SET)

        # Keep reading to find the exact target date
        sice0 = sice.copy()

        while True:
            try:
                hdr = f.read_reals(dtype=np.float32)
                sice = f.read_reals(dtype=np.float32).reshape((nx, ny))

                cur_y = int(hdr[0])
                cur_m = int(hdr[1])
                cur_d = int(hdr[2])

                # Check if we reached the target date
                if cur_y == year and cur_m == month and cur_d == day:
                    break
                else:
                    sice0 = sice.copy()  # Store as previous day's info

            except Exception as e:
                print(
                    f"Reached EOF or encountered an error before finding {target_date.date()}"
                )
                break

    # Interpolate based on the hour
    weight = hour / 24.0
    sice_interp = sice * weight + sice0 * (1.0 - weight)

    return sice_interp, hdr
