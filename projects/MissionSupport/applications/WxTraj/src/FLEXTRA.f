      program flextra
********************************************************************************
*                                                                              *
*     This program calculates trajectories for various input wind fields.      *
*                                                                              *
*     Authors: A. Stohl, G. Wotawa                                             *
*                                                                              *
*     2 February 1994                                                          *
*                                                                              *
*     Update: January 1999: A. Stohl                                           *
*     Use of global fields, CET option, etc.                                   *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* error                .true., if error ocurred in subprogram, else .false.    *
*                                                                              *
* Constants:                                                                   *
*                                                                              *
********************************************************************************
      use netcdf
      include 'includepar'
      include 'includecom'
      include 'netcdf.inc'

      logical error

C Read the command line argument for the pathnames filename
************************************************************

      call getarg(1,pathnames)


C Read the pathnames where input/output files are stored
********************************************************

      call readpaths(error)
      if (error) goto 999


C Read the user specifications for the current model run
********************************************************

      call readcommand(error)
      if (error) goto 999

C Read, which wind fields are available within the modelling period
*******************************************************************

      call readavailable(error)
      if(error) goto 999

C Determine the grid specifications, vertical discretization, and orography
***************************************************************************
       
      call gridcheck(error)
      if(error) goto 999
!      call gridcheck_nests(error)   ! PC not implementing nests for now
!      if(error) goto 999
       
 

C Read the coordinates of trajectory beginning/ending points for the
C current model run
C Alternatively, if CET is to be calculated read CET starting domain
********************************************************************

      if (modecet.eq.1) then
        call readpoints(error)
      else if (modecet.eq.2) then
        call readcet(error)
      else
        call readflight(error)
      endif
      if(error) goto 999
      

!C Check, if user selected options don't exceed the current dimension limits
***************************************************************************

      call checklimits(error)
      if(error) goto 999
      
C Conversion of the startpoints from geografical to grid coordinates
********************************************************************

      call coordtrafo(error)
      if(error) goto 999
      

C Fix the coordinates of the uncertainty trajectories
*****************************************************

      if (modecet.eq.1) call uncertcoor()

 
C Subtract the orography from the height above sea level
********************************************************

      call subtractoro()
      

C Open the output files
***********************

      if (modecet.eq.1) then
        call openoutput(error)
      else if (modecet.eq.2) then
        call opencetoutput(error)
      else
        call openflightoutput(error)
      endif
      if(error) goto 999


C Calculate trajectories
************************
      
      call timemanager()


C Close output and reverse direction of back trajectory output
**************************************************************

      if (modecet.eq.1) call lastprocessor()

      write(*,*) 'CONGRATULATIONS: YOU HAVE SUCCESSFULLY COMPLETED A  FL
     +EXTRA  MODEL RUN!'

      goto 1000

999   write(*,*) 'FLEXTRA MODEL ERROR: EXECUTION HAD TO BE TERMINATED'

1000  continue

      end
