# US-View City Selection Summary

## What the script does

- Script reviewed: `links/IDL_BASE/ploteic_t2m.pro`.
- For US-only (`eicusa_mapset`), city candidates come from `/home/wputman/IDL_BASE/CITIES/world_cities.csv`.
- Cities are projected to device coordinates, clipped to map bounds, then thinned by overlap suppression (`imarkCities`).
- The plotted label is temperature text at each selected city location.

## Map bounds used

- Region: `REGION=104` (`eicusa_mapset` in `setup_region.pro`).
- Latitude: 23.0 to 50.0
- Longitude: -125.75 to -68.25

## Outputs written

- City-by-city CSV: `us_view_city_list.csv`

## City counts in output CSV

- Total selected cities: 536
- United States of America: 417
- Canada: 60
- Mexico: 55
- Cuba: 2
- The Bahamas: 2

## Notes

- `state_if_us` is populated only for rows where country is `United States of America`.
- File order follows `world_cities.csv` order after filtering and overlap suppression.
