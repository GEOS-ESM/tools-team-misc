      subroutine readcommand(error)
C                              o
********************************************************************************
*                                                                              *
*     This routine reads the user specifications for the current model run.    *
*                                                                              *
*     Authors: A. Stohl                                                        *
*                                                                              *
*     2 February 1994                                                          *
*                                                                              *
*     10 January 1999  Update to facilitate free formatted input               *
*     (P. Seibert + A. Stohl)                                                  *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* bdate                beginning date as Julian date                           *
* edate                ending date as Julian date                              *
* error                .true., if error ocurred in subprogram, else .false.    *
* hhh                  hour                                                    *
* ibdate,ibtime        beginnning date and time (YYYYMMDD, HHMISS)             *
* ideltas [s]          modelling period                                        *
* iedate,ietime        ending date and time (YYYYMMDD, HHMISS)                 *
* interv [s]           interval between two trajectory calculations            *
* ldirect              -1 for backward, 1 for forward trajectories             *
* lentra [s]           length of one trajectory                                *
* line                 a line of text                                          *
* ldim                 number of steps along the interpolated trajectory       *
* mi                   minute                                                  *
* relaxtime            time constant at which random errors are relaxed        *
* ss                   second                                                  *
*                                                                              *
* Constants:                                                                   *
* unitcommand          unit connected to file COMMAND                          *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      logical error,old
      integer hhh,mi,ss
      double precision edate,juldate
      character*50 line


C Open the command file and read user options
*********************************************

      error=.false.

      open(unitcommand,file=path(1)(1:len(1))//'COMMAND',status='old',
     +err=999)

C Check the format of the COMMAND file (either in free format,
C or using formatted mask)
C Use of formatted mask is assumed if line 9 contains the word 'LABEL'
**********************************************************************

      call skplin(8,unitcommand)
      read (unitcommand,901) line
901   format (a)
      if (index(line,'LABEL') .eq. 0) then
        old = .false.
      else
        old = .true.
      endif
      rewind(unitcommand)

C Read parameters
*****************

      call skplin(6,unitcommand)
      if (old) call skplin(1,unitcommand)
      if (old) then
        read(unitcommand,'(3x,a50)') runcomment
      else
        read(unitcommand,*) runcomment
      endif
      if (old) call skplin(3,unitcommand)
      read(unitcommand,*) ldirect
      if (old) call skplin(3,unitcommand)
      read(unitcommand,*) lentra
      if (old) call skplin(3,unitcommand)
      read(unitcommand,*) ibdate,ibtime
      if (old) call skplin(3,unitcommand)
      read(unitcommand,*) iedate,ietime
      if (old) call skplin(3,unitcommand)
      read(unitcommand,*) interv         
      interv=max(interv,1)         ! minimum interval 1 second
      if (old) call skplin(3,unitcommand)
      read(unitcommand,*) inter,interstep
      if (old) call skplin(3,unitcommand)
      read(unitcommand,*) numbunc,distunc,relaxtime,epsu,epsv,epsw
      if (old) call skplin(3,unitcommand)
      read(unitcommand,*) inpolkind
      if (old) call skplin(3,unitcommand)
      read(unitcommand,*) cfl
      if (old) call skplin(3,unitcommand)
      read(unitcommand,*) cflt
      if (old) call skplin(3,unitcommand)
      read(unitcommand,*) modecet
      
      close(unitcommand)


C Check input dates
*******************

      if (iedate.lt.ibdate) then
        write(*,*) ' #### TRAJECTORY MODEL ERROR! BEGINNING DATE  #### ' 
        write(*,*) ' #### IS LARGER THAN ENDING DATE. CHANGE      #### '
        write(*,*) ' #### EITHER POINT 3 OR POINT 4 IN FILE       #### '
        write(*,*) ' #### "COMMAND".                              #### '
        error=.true.
        return
      else if (iedate.eq.ibdate) then
        if (ietime.lt.ibtime) then
        write(*,*) ' #### TRAJECTORY MODEL ERROR! BEGINNING TIME  #### ' 
        write(*,*) ' #### IS LARGER THAN ENDING TIME. CHANGE      #### '
        write(*,*) ' #### EITHER POINT 3 OR POINT 4 IN FILE       #### '
        write(*,*) ' #### "COMMAND".                              #### '
        error=.true.
        return

        endif
      endif


C Check CFL criterions
**********************

      if((cfl.lt.1.).or.(cflt.lt.1.)) then
        write(*,*) ' #### TRAJECTORY MODEL ERROR! CFL CRITERION   #### ' 
        write(*,*) ' #### MUST NOT BE SET LESS THEN 1 !!!         #### '
        error=.true.
        return
      endif


C Conversion of format HHHMISS to seconds
*****************************************

      hhh=lentra/10000
      mi=(lentra-10000*hhh)/100
      ss=lentra-10000*hhh-100*mi
      lentra=ldirect*(hhh*3600+60*mi+ss)

      hhh=interv/10000
      mi=(interv-10000*hhh)/100
      ss=interv-10000*hhh-100*mi
      interv=hhh*3600+60*mi+ss


C Compute number of time steps along interpolated trajectory
************************************************************

      if (interstep.lt.1) interstep=1
      ldim=ldirect*lentra/interstep+1

      if ((inter.ge.0).and.(ldim.gt.maxitime)) then
        write(*,*) ' #### TRAJECTORY MODEL ERROR! INTERPOLATION   #### ' 
        write(*,*) ' #### TO SUCH A SHORT TIME STEP IS NOT        #### '
        write(*,*) ' #### POSSIBLE. SET SSSSS IN POINT 7. OF FILE #### '
        write(*,*) ' #### "COMMAND" TO A GREATER VALUE.           #### '
        error=.true.
        return
      endif

      if ((inter.eq.1).and.(maxtime.lt.2*ldim+1)) then
        write(*,*) ' #### TRAJECTORY MODEL ERROR! INTERPOLATION   #### ' 
        write(*,*) ' #### TO SUCH A SHORT TIME STEP IS NOT        #### '
        write(*,*) ' #### SENSIBLE GIVEN CURRENT SETTING OF       #### '
        write(*,*) ' #### MAXTRA. SET SSSSS IN POINT 7. OF FILE   #### '
        write(*,*) ' #### "COMMAND" TO A GREATER VALUE.           #### '
        error=.true.
        return
      endif

C Compute modelling time in seconds and beginning date in Julian date
*********************************************************************

      if (ldirect.eq.1) then
        bdate=juldate(ibdate,ibtime)
        edate=juldate(iedate,ietime)
        ideltas=nint(86400.*(edate-bdate))+lentra
      else if (ldirect.eq.-1) then
        bdate=juldate(iedate,ietime)
        edate=juldate(ibdate,ibtime)
        ideltas=nint(86400.*(edate-bdate))+lentra
      else
        write(*,*) ' #### TRAJECTORY MODEL ERROR! DIRECTION IN    #### ' 
        write(*,*) ' #### FILE "COMMAND" MUST BE EITHER -1 OR 1.  #### '
        error=.true.
        return
      endif


      return    

999   write(*,*) ' #### TRAJECTORY MODEL ERROR! FILE "COMMAND"  #### ' 
      write(*,*) ' #### CANNOT BE OPENED IN THE DIRECTORY       #### '
      write(*,*) ' #### xxx/trajec/options                      #### '
      error=.true.

      return
      end
