import h5py
import numpy as np

from .observation import Observation
from .registry import register


@register("OMI_Aura_L2_OMAERO")
class OMI_Aura_L2_OMAERO(Observation):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.obs_category = "Polar Orbiting"
        self.legend_category = "Aura OMI"
        self.isis = "none"

        self.set_grid_method("h3", 3)

    def read(self, filename):

        base_epoch = np.datetime64("1993-01-01T00:00:00", "s")

        with h5py.File(filename, "r") as fh:

            swath = fh["HDFEOS/SWATHS/ColumnAmountAerosol"]
            geo = swath["Geolocation Fields"]
            data = swath["Data Fields"]

            # Read geolocation data
            lats = geo["Latitude"][:]
            lons = geo["Longitude"][:]
            times = geo["Time"][:]
            sza = geo["SolarZenithAngle"][:]
            gp_flags = geo["GroundPixelQualityFlags"][:]

            # Read quality and cloud flags
            xtrack_flags = data["XTrackQualityFlags"][:]
            meas_flags = data["MeasurementQualityFlags"][:]
            proc_flags = data["ProcessingQualityFlagsMW"][:]
            cloud_frac = data["EffectiveCloudFraction"][:] * 0.01

            # Repeat row data for all columns.
            num_cols = lats.shape[1]
            times = np.repeat(times[:, np.newaxis], num_cols, axis=1)
            times = np.round(times).astype("int64").astype("timedelta64[s]") + base_epoch
            meas_flags = np.repeat(meas_flags[:, np.newaxis], num_cols, axis=1)

            # Build combined high-quality boolean mask
            valid_mask = (
              (xtrack_flags == 0)  # No row anomaly
            & (meas_flags == 0)  # Good measurement quality
            & ((proc_flags & 15) == 0)  # No fatal MW retrieval errors (bits 0-3 == 0)
            & (cloud_frac >= 0)  # Valid cloud fraction (not fill value -127)
            & (cloud_frac <= 10)  # Effective cloud fraction <= 0.10
            & (sza < 75.0)  # Reasonable solar zenith angle
            & ((gp_flags & 16) == 0)  # Bit 4: exclude sun glint over ocean
)

            self.lats = lats[valid_mask].flatten()
            self.lons = lons[valid_mask].flatten()
            self.times = times[valid_mask].flatten()
