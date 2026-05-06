      subroutine readoro(error)
C                          o
********************************************************************************
*                                                                              *
*     This routine reads the orography used by the ECMWF model.                *
*                                                                              *
*     Authors: A. Stohl                                                        *
*                                                                              *
*     30 April 1994                                                            *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* error                .true., if file OROGRAPHY cannot be opened              *
* nx,ny                number of grid points in x and y direction, respectively*
* oro(0:nxmax-1,0:nymax-1) [m]  orography of ECMWF model                       *
*                                                                              *
* Constants:                                                                   *
* unitoro              unit connected to file OROGRAPHY                        *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer ix,jy
      logical error


      error=.false.

C Open the orography file and read grid
***************************************

      write(*,*) 'NOTICE: OROGRAPHY WAS READ FROM FILE "OROGRAPHY"'


      open(unitoro,file=path(3)(1:len(3))//'OROGRAPHY',
     +status='old',err=999)
      do 10 ix=0,nx-1
        do 10 jy=0,ny-1
10        read(unitoro,*) oro(ix,jy)
      close(unitoro)

      return    

999   write(*,*) ' #### FLEXTRA MODEL ERROR! FILE "OROGRAPHY"   #### ' 
      write(*,*) ' #### CANNOT BE OPENED IN THE DIRECTORY       #### '
      write(*,*) ' #### xxx/trajec/windfields                   #### '
      error=.true.

      return
      end
