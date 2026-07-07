      subroutine readavailable(error)
C                                o
********************************************************************************
*                                                                              *
*   This routine reads the dates and times for which windfields are available. *
*                                                                              *
*     Authors: A. Stohl                                                        *
*                                                                              *
*     6 February 1994                                                          *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* bdate                beginning date as Julian date                           *
* beg                  beginning date for windfields                           *
* end                  ending date for windfields                              *
* error                .true., if error ocurred in subprogram, else .false.    *
* fname                filename of wind field, help variable                   *
* ideltas [s]          duration of modelling period                            *
* idiff                time difference between 2 wind fields                   *
* idiffnorm            normal time difference between 2 wind fields            *
* idiffmax [s]         maximum allowable time between 2 wind fields            *
* jul                  julian date, help variable                              *
* ldirect              -1 for backward, 1 for forward trajectories             *
* numbnests            actual number of nest levels                            *
* numbwf               actual number of wind fields                            *
* wfname(numbwfmax)    file names of needed wind fields                        *
* wfspec(numbwfmax)    file specifications of wind fields (e.g., if on disc)   *
* wftime(numbwfmax) [s]times of wind fields relative to beginning time         *
* wfname1,wfspec1,wftime1 = same as above, but only local (help variables)     *
*                                                                              *
* Constants:                                                                   *
* numbwfmax            maximum number of wind fields                           *
* unitavailab          unit connected to file AVAILABLE                        *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      logical error
      integer i,k,idiff,ldat,ltim,wftime1(numbwfmax),numbwfn(maxnests)
      integer wftime1n(maxnests,numbwfmax),wftimen(maxnests,numbwfmax)
      double precision juldate,jul,beg,end
      character*180 fname
      character*10 spec, wfspec1(numbwfmax)
      character*180 wfname1(numbwfmax)
      character*18 wfname1n(maxnests,numbwfmax)
      character*10 wfspec1n(maxnests,numbwfmax)
     


      error=.false.


C Windfields are only used, if they are within the modelling period.
C However, 1 additional day at the beginning and at the end is used for
C interpolation. -> Compute beginning and ending date for the windfields.
*************************************************************************

      if (ideltas.gt.0) then         ! forward trajectories
        beg=bdate-1.                  
        end=bdate+dble(float(ideltas)/86400.)+dble(float(idiffmax)/
     +  86400.)
      else                           ! backward trajectories
        beg=bdate+dble(float(ideltas)/86400.)-dble(float(idiffmax)/
     +  86400.)
        end=bdate+1.
      endif

C Open the wind field availability file and read available wind fields
C within the modelling period (mother grid)
**********************************************************************

      open(unitavailab,file=path(4)(1:len(4)),status='old',
     +err=999)

      do i=1,3
        read(unitavailab,*)
      end do
      
      numbwf=0
100     read(unitavailab,'(i8,1x,i6,2(6x,a180))',end=99) ldat,ltim,
     +  fname,spec
        
        jul=juldate(ldat,ltim)
        if ((jul.ge.beg).and.(jul.le.end)) then
          numbwf=numbwf+1
          if (numbwf.gt.numbwfmax) then      ! check exceedance of dimension
           write(*,*) 'Number of wind fields needed is too great.'
           write(*,*) 'Reduce modelling period (file "COMMAND") or'
           write(*,*) 'reduce number of wind fields (file "AVAILABLE").'
           goto 1000
          endif

          wfname1(numbwf)=fname(1:index(fname,' '))
          wfspec1(numbwf)=spec
          wftime1(numbwf)=nint((jul-bdate)*86400.)
        endif
        goto 100       ! next wind field

99    continue

      close(unitavailab)


C Open the wind field availability file and read available wind fields
C within the modelling period (nested grids)
**********************************************************************

      do k=1,numbnests
        open(unitavailab,file=path(numpath+2*(k-1)+2)
     +  (1:len(numpath+2*(k-1)+2)),status='old',err=998)

        do i=1,3
          read(unitavailab,*)
        end do
      
        numbwfn(k)=0
700       read(unitavailab,'(i8,1x,i6,2(6x,a18))',end=699) ldat,
     +    ltim,fname,spec
          jul=juldate(ldat,ltim)
          if ((jul.ge.beg).and.(jul.le.end)) then
            numbwfn(k)=numbwfn(k)+1
            if (numbwfn(k).gt.numbwfmax) then      ! check exceedance of dimension
           write(*,*) 'Number of nested wind fields is too great.'
           write(*,*) 'Reduce modelling period (file "COMMAND") or'
           write(*,*) 'reduce number of wind fields (file "AVAILABLE").'
              goto 1000
            endif

            wfname1n(k,numbwfn(k))=fname
            wfspec1n(k,numbwfn(k))=spec
            wftime1n(k,numbwfn(k))=nint((jul-bdate)*86400.)
          endif
          goto 700       ! next wind field

699     continue

        close(unitavailab)
        end do

C Check wind field times of file AVAILABLE (expected to be in temporal order)
*****************************************************************************

      if (numbwf.eq.0) then
        write(*,*) ' #### TRAJECTORY MODEL ERROR! NO WIND FIELDS  #### ' 
        write(*,*) ' #### AVAILABLE FOR SELECTED TIME PERIOD.     #### '
        error=.TRUE.
        return
      endif

      do 150 i=2,numbwf
        if (wftime1(i).le.wftime1(i-1)) then
          write(*,*) 'FLEXTRA ERROR: FILE AVAILABLE IS CORRUPT.'
          write(*,*) 'THE WIND FIELDS ARE NOT IN TEMPORAL ORDER.'
          write(*,*) 'PLEASE CHECK FIELD ',wfname1(i)
          error=.TRUE.
          return
        endif
150     continue


C Check wind field times of file AVAILABLE (expected to be in temporal order)
*****************************************************************************

      do 77 k=1,numbnests
        if (numbwfn(k).eq.0) then
        write(*,*) ' #### TRAJECTORY MODEL ERROR! NO WIND FIELDS  #### ' 
        write(*,*) ' #### AVAILABLE FOR SELECTED TIME PERIOD.     #### '
          error=.TRUE.
          return
        endif

        do 160 i=2,numbwfn(k)
          if (wftime1n(k,i).le.wftime1n(k,i-1)) then
          write(*,*) 'FLEXTRA ERROR: FILE AVAILABLE IS CORRUPT.'
          write(*,*) 'THE NESTED WIND FIELDS ARE NOT IN TEMPORAL ORDER.'
          write(*,*) 'PLEASE CHECK FIELD ',wfname1n(k,i)
          write(*,*) 'AT NESTING LEVEL ',k
            error=.TRUE.
            return
          endif
160     continue

77      continue


C For backward trajectories, reverse the order of the windfields
****************************************************************

      if (ideltas.ge.0) then
        do i=1,numbwf
          wfname(i)=wfname1(i)
          wfspec(i)=wfspec1(i)
          wftime(i)=wftime1(i)
        end do
        do k=1,numbnests
          do i=1,numbwfn(k)
            wfnamen(k,i)=wfname1n(k,i)
            wfspecn(k,i)=wfspec1n(k,i)
            wftimen(k,i)=wftime1n(k,i)
          end do
        end do
      else
        do i=1,numbwf
          wfname(numbwf-i+1)=wfname1(i)
          wfspec(numbwf-i+1)=wfspec1(i)
          wftime(numbwf-i+1)=wftime1(i)
        end do
        do k=1,numbnests
          do i=1,numbwfn(k)
            wfnamen(k,numbwfn(k)-i+1)=wfname1n(k,i)
            wfspecn(k,numbwfn(k)-i+1)=wfspec1n(k,i)
            wftimen(k,numbwfn(k)-i+1)=wftime1n(k,i)
          end do
        end do
      endif


C Check the time difference between the wind fields. If it is big, 
C write a warning message. If it is too big, terminate the trajectory.
**********************************************************************

      do 350 i=2,numbwf
        idiff=abs(wftime(i)-wftime(i-1))
        if (idiff.gt.idiffmax) then
          write(*,*) 'FLEXTRA WARNING: TIME DIFFERENCE BETWEEN TWO'
          write(*,*) 'WIND FIELDS IS TOO BIG FOR TRAJECTORY CALCULATION.
     &               '
          write(*,*) 'THEREFORE, TRAJECTORIES HAVE TO BE SKIPPED.'
        else if (idiff.gt.idiffnorm) then
          write(*,*) 'FLEXTRA WARNING: TIME DIFFERENCE BETWEEN TWO'
          write(*,*) 'WIND FIELDS IS BIG. THIS MAY CAUSE A DEGRADATION'
          write(*,*) 'OF TRAJECTORY QUALITY.'
        endif
350     continue

      do k=1,numbnests
        if (numbwfn(k).ne.numbwf) then
          write(*,*) 'FLEXTRA ERROR: THE AVAILABLE FILES FOR THE'
          write(*,*) 'NESTED WIND FIELDS ARE NOT CONSISTENT WITH'
          write(*,*) 'THE AVAILABLE FILE OF THE MOTHER DOMAIN.  '
          write(*,*) 'ERROR AT NEST LEVEL: ',k
          goto 1000
        endif
        do i=1,numbwf
          if (wftimen(k,i).ne.wftime(i)) then
            write(*,*) 'FLEXTRA ERROR: THE AVAILABLE FILES FOR THE'
            write(*,*) 'NESTED WIND FIELDS ARE NOT CONSISTENT WITH'
            write(*,*) 'THE AVAILABLE FILE OF THE MOTHER DOMAIN.  '
            write(*,*) 'ERROR AT NEST LEVEL: ',k
            goto 1000
          endif
        end do
      end do


C Reset the times of the wind fields that are kept in memory to no time
***********************************************************************

      do i=1,3
        memind(i)=i
        memtime(i)=999999999
      end do
      return    

998   write(*,*) ' #### TRAJECTORY MODEL ERROR! FILE #### '
      write(*,'(a)') '     '//path(numpath+2*(k-1)+2)
     +(1:len(numpath+2*(k-1)+2))
      write(*,*) ' #### CANNOT BE OPENED             #### '
      error=.true.
      return

999   write(*,*) ' #### TRAJECTORY MODEL ERROR! FILE #### '
      write(*,'(a)') '     '//path(4)(1:len(4))
      write(*,*) ' #### CANNOT BE OPENED             #### '
1000  error=.true.

      return
      end
