#!/bin/csh


source ./g5_modules

#set YEAR_TABLE  = ( 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 )
#set MONTH_TABLE = ( 01 02 03 04 05 06 07 08 09 10 11 12 )

set YEAR_TABLE  = ( 2019 )
set MONTH_TABLE = ( 02 )

#set DAY_TABLE   = ( 01 02 03 04 05 )
#set DAY_TABLE   = ( 06 07 08 09 10 )
#set DAY_TABLE   = ( 11 12 13 14 15 )
#set DAY_TABLE   = ( 16 17 18 19 20 )
#set DAY_TABLE   = ( 21 22 23 24 25 )
#set DAY_TABLE   = ( 26 27 28 )
#set DAY_TABLE   = ( 26 27 28 29 30 31)
set DAY_TABLE    = ( 19 22 28 )

set ALT_TABLE   = ( 100 )
#set ALT_TABLE   = ( 1000 )

foreach YYYY ( `echo $YEAR_TABLE` )
    foreach MM ( `echo $MONTH_TABLE` )
        foreach DD ( `echo $DAY_TABLE` )
            set ISODATE   =  ${YYYY}-${MM}-${DD}T00:00:00
            foreach AA ( `echo $ALT_TABLE` )
                echo 'nohup python3 -u ./flextra_run_ops.py --alts='${AA}' --days=5 '${ISODATE}' start_points.csv >& nohup.'${ISODATE}_${AA}'.log'
                nohup python3 -u ./flextra_run_ops.py  --days=5  ${ISODATE} start_points.csv >& nohup.${ISODATE}_${AA}.log &
            end
        end
    end
end
  
