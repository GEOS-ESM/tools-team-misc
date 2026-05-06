      subroutine getfields(firststep,itime,indexf,idiff,nstop)
C                              i       i     o      o     o
********************************************************************************
*                                                                              *
*  This subroutine manages the 3 data fields to be kept in memory.             *
*  During the first time step of petterssen it has to be fulfilled that the    *
*  first data field must have |wftime|<itime, i.e. the absolute value of wftime*
*  must be smaller than the absolute value of the current time in [s].         *
*  The other 2 fields are the next in time after the first one.                *
*  Pointers (memind) are used, because otherwise one would have to resort the  *
*  wind fields, which costs a lot of computing time. Here only the pointers are*
*  resorted.                                                                   *
*                                                                              *
*     Author: A. Stohl                                                         *
*                                                                              *
*     29 April 1994                                                            *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* firststep            .true. for first time step of petterssen, else .false.  *
* idiff [s]            time difference between the two wind fieldse read in    *
* indj                 indicates the number of the wind field to be read in    *
* indmin               remembers the number of wind fields already treated     *
* memind(3)            pointer, on which place the wind fields are stored      *
* memtime(3) [s]       times of the wind fields, which are kept in memory      *
* itime [s]            current time since start date of trajectory calculation *
* ldirect              1 for forward trajectories, -1 for backward trajectories*
* nstop                > 0, if trajectory has to be terminated                 *
* nx,ny,nuvz,nwz       field dimensions in x,y and z direction                 *
* uu(0:nxmax-1,0:nymax-1,nuvzmax,3)   wind components in x-direction [m/s]     *
* vv(0:nxmax-1,0:nymax-1,nuvzmax,3)   wind components in y-direction [m/s]     *
* ww(0:nxmax-1,0:nymax-1,nwzmax,3)wind components in z-direction [deltaeta/s]  *
* tt(0:nxmax-1,0:nymax-1,nuvzmax,3)   temperature [K]                          *
* ps(0:nxmax-1,0:nymax-1,3)           surface pressure [Pa]                    *
*                                                                              *
* Constants:                                                                   *
* idiffnorm            normal time difference between 2 wind fields            *
* idiffmax             maximum allowable time difference between 2 wind fields *
*                                                                            *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer indj,indmin,l,itime,indexf,idiff,nstop,memhelp
      logical firststep
      save indmin

      data indmin/1/

 
C Check, if wind fields are available for the current time step
***************************************************************

      if ((ldirect*wftime(1).ge.ldirect*itime).or.
     +(ldirect*wftime(numbwf).le.ldirect*itime)) then
        write(*,*) 'FLEXTRA WARNING: NO WIND FIELDS ARE AVAILABLE.'
        write(*,*) 'A TRAJECTORY HAS TO BE TERMINATED.'
        nstop=4
        return
      endif


******************************************************************************
C For the first time step of petterssen, arrange the wind fields in such a way
C that 1st wind field is before itime and 2nd and 3rd are after itime.
******************************************************************************

      if (firststep) then
        if ((ldirect*memtime(1).lt.ldirect*itime).and.
     +  (ldirect*memtime(2).ge.ldirect*itime)) then

! The right wind fields are already in memory -> don't do anything
******************************************************************

          continue

        else if ((ldirect*memtime(2).lt.ldirect*itime).and.
     +  (ldirect*memtime(3).ge.ldirect*itime)) then


C Current time is between 2nd and 3rd wind field
C -> Resort wind field pointers, so that current time is between 1st and 2nd
****************************************************************************

          memhelp=memind(1)
          do l=1,2
            memind(l)=memind(l+1)
            memtime(l)=memtime(l+1)
          end do
          memind(3)=memhelp


C Read a new wind field and store it on place memind(3)
*******************************************************

          do 30 indj=indmin,numbwf-2
            if ((ldirect*wftime(indj).lt.ldirect*itime).and.
     +      (ldirect*wftime(indj+1).ge.ldirect*itime)) then
              call readwind(indj+2,memind(3))
!              call readwind_nests(indj+2,memind(3))
              memtime(3)=wftime(indj+2)
              goto 40
            endif
30          continue
40        indmin=indj

        else

C No wind fields, which can be used, are currently in memory 
C -> read all 3 wind fields
************************************************************

          do 50 indj=indmin,numbwf-1
            if ((ldirect*wftime(indj).lt.ldirect*itime).and.
     +      (ldirect*wftime(indj+1).ge.ldirect*itime)) then
              memind(1)=1
              call readwind(indj,memind(1))
!              call readwind_nests(indj,memind(1))
              memtime(1)=wftime(indj)
              memind(2)=2
              call readwind(indj+1,memind(2))
!              call readwind_nests(indj+1,memind(2))
              memtime(2)=wftime(indj+1)
              memind(3)=3
              call readwind(indj+2,memind(3))
!              call readwind_nests(indj+2,memind(3))
              memtime(3)=wftime(indj+2)
              goto 60
            endif
50          continue
60        indmin=indj

        endif

        indexf=1
        idiff=abs(memtime(2)-memtime(1))


*****************************************************************************
C For 2nd step of petterssen all necessary data fields are already in memory.
C Just look, if current temporal position is between the first two fields or
C between the 2nd and 3rd field.
*****************************************************************************

      else
        if ((ldirect*memtime(1).lt.ldirect*itime).and.    !between 1st and 2nd
     +  (ldirect*memtime(2).ge.ldirect*itime)) then
          indexf=1
          idiff=abs(memtime(2)-memtime(1))
        else if ((ldirect*memtime(2).lt.ldirect*itime).and. !between 2nd and 3rd
     +  (ldirect*memtime(3).ge.ldirect*itime)) then
          indexf=2
          idiff=abs(memtime(3)-memtime(2))
        endif
      endif

C Check the time difference between the wind fields. If it is too 
C big, terminate the trajectory. 
******************************************************************

      if (idiff.gt.idiffmax) nstop=3

      return
      end
