# EarthNow Products
For IDL script references, look in this dir:
`/home/wputman/IDL_BASE`

## Radiation band variables
`ploteic_geocolor`
GeoColor: True View of Earth
- Assignee: Emily
- Status:

`ploteic_sandwich`
Sandwich RGB - Blended LW IR and Veggie VIS
- Assignee:
- Status:

`ploteic_band16`
13.3 micron - CO2 Longwave Band IR
- Assignee:
- Status:

`ploteic_band14`
11.2 micron - Longwave Band IR
- Assignee: Bennett
- Status:

`ploteic_band09`
6.9 micron - Mid Level Water Vapor IR
- Assignee: Bennett
- Status:


## Surface variables
`ploteic_radar`
Radar Reflectivity [Rain/Snow/Ice]
- Assignee: Emily
- Status:

`ploteic_precip`
Accumulated Precip [Rain & Snow]
- Assignee: Emily
- Status: Done!

`ploteic_winds`
Near Surface Winds
- Assignee: Emily
- Status: Done!

`ploteic_t2m`
2-meter Temperature
- Assignee: Sandra
- Status: Done!

`ploteic_tanom`
2-meter Temperature Anomaly (1-day running mean)
- Assignee:

`ploteic_cape`
Convective Available Potential Energy
- Assignee: Hannah
- Status:

`ploteic_helicity`
2-5km Helicity and Radar Reflectivity
- Assignee: Hannah
- Status: Complete but do not have existing animation to compare (see emails with Bill)
- Function call: plotall.py --product "max_reflectivity_EarthNow" --style "helicity"


## Composition
`ploteic_aerosols`
Aerosol Optical Thickness [SS, DU, SU, NI]
- Assignee: Sandra / Hannah
- Status:

`ploteic_carbon`
Carbon Aerosol Optical Thickness
- Assignee: Sandra
- Status:


## Global Dynamics
`ploteic_vort500`
500-mb Vorticity and Heights
- Assignee:Hannah
- Status: Done! Except for locating the identical basemap that the IDL code uses.
- Function call: plotall.py --product "vorticity_heights_500mb_EarthNow"  --style "grey_topo"

`ploteic_wind250`
250-mb Wind Speed and MSLP
- Assignee: Emily
- Status: Done!
