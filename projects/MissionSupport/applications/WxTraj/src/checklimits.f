      subroutine checklimits(error)       
C                              o
********************************************************************************
*                                                                              *
*     This routine checks, if the current user specifications are within the   *
*     dimensional limits of the number of trajectories that have to be handled *
*     at the same time.                                                        *
*                                                                              *
*     Author: A. Stohl                                                         *
*                                                                              *
*     2 February 1994                                                          *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* error                  .true., if dimensions are exceeded, else .false.      *
* interv [s]             interval between two trajectory calculations          *
* lentra [s]             length of an individual trajectory                    *
* numtramax              wanted maximum number of trajectories                 *
*                                                                              *
* Constants:                                                                   *
* maxtra                 maximum allowable number of trajectories              *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer numtramax
      logical error

      error=.false.

1     format(1x,a6,i7,a43)

C Calculate maximum number of trajectories to be held in memory 
C If only one starting time is selected, numtramax=numpoint, otherwise it is
C =(number of starting points)*(length of trajectories)/(interval of starting
C times of trajectories)
*****************************************************************************

      if ((ibdate.eq.iedate).and.(ibtime.eq.ietime)) then
        numtramax=numpoint
      else
        numtramax=numpoint*int(float(abs(lentra))/float(interv)+1.)
      endif

      if (modecet.eq.3) then
        numtramax=1
      endif


C If numtramax is greater than the number allowed by maxtra, give a warning
C and stop the model execution.
***************************************************************************

      if (numtramax.gt.maxtra) then
        error=.true.
        write(*,*) '#####                !!!ERROR!!!                   #
     +####'
        write(*,*) '##### YOU WANT TO CALCULATE TOO MANY TRAJECTORIES: #
     +####'
        write(*,1) '##### ',numtramax,' ARE WANTED, BUT ONLY                 
     +     #####'
        write(*,1) '##### ',maxtra,' ARE POSSIBLE.                     
     +  #####'
        write(*,*) '##### YOU CAN AVOID THIS PROBLEM IN THREE WAYS:    #
     +####'
        write(*,*) '##### 1) REDUCE THE NUMBER OF STARTING POINTS IN   #
     +####'
        write(*,*) '#####    FILE "STARTPOINTS".                       #
     +####'
        write(*,*) '##### 2) REDUCE THE CALCULATION LENGTH OF THE      #
     +####'
        write(*,*) '#####    TRAJECTORIES IN FILE "COMMAND".           #
     +####'
        write(*,*) '##### 3) INCREASE THE TIME INTERVAL BETWEEN TWO    #
     +####'
        write(*,*) '#####    TRAJECTORY CALCULATIONS IN FILE "COMMAND".#
     +####'
      endif

      return
      end
