#!/bin/sh

while read filename; do

    field=`echo $filename | cut -d'_' -f1`
    airport=`echo $filename | cut -d'_' -f2`
    aircraft=`echo $filename | cut -d'_' -f3`
    departure=`echo $filename | cut -d'_' -f4 | cut -d'.' -f1`

    oname=nasa.gmao.flight_track.$aircraft.GEOSFP.$field.0.$airport.$departure.PT000H.png

    mv $filename $oname

done

exit 0
