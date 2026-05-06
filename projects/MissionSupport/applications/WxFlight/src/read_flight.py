import os
import sys
import glob
from netCDF4 import Dataset

from flight import *

fl = Flight(sys.argv[1])

flights = fl.get_flights()

#flight = { 'location': 'Manila', 'aircraft': 'DC8' }
#field = get_data(sys.argv[1], 'BC', flight)

#print(field['time'])

for k,v in flights.items():
    data = fl.get_data('BC' , v)
    print(k)
    print(v['airport'])
    print(v['aircraft'])
    print(v['departure'])

    waypoints = fl.get_waypoints(v)
    print(waypoints)
