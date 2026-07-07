      subroutine readflight(error)
C                             o
***********************************************************************
*                                                                     * 
*             TRAJECTORY MODEL SUBROUTINE READCET                     *
*                                                                     *
***********************************************************************
*                                                                     * 
*             AUTHOR:      A. STOHL                                   *
*             DATE:        1999-02-01                                 *
*                                                                     * 
* Update: Vertical coordinate can now be given in meters above sea    * 
* level, meters above ground, and in hPa.                             * 
*                                                                     * 
***********************************************************************
*                                                                     *
* DESCRIPTION:                                                        *
*                                                                     *
* READING OF TRAJECTORY STARTING/ENDING POINTS FROM DATA FILE         *
*                                                                     * 
* LINE                  a line of text                                * 
* NUMPOINT              number of trajectory starting/ending points   *
* XPOINT(maxpoint)      x-coordinates of starting/ending points       *
* YPOINT(maxpoint)      y-coordinates of starting/ending points       *
* ZPOINT(maxpoint)      z-coordinates of starting/ending points       *
* KINDZ(maxpoint)       kind of z coordinate (1:masl, 2:magl, 3:hPa)  *
* KIND(maxpoint)        kind of trajectory                            *
* COMPOINT(maxpoint)    comment for trajectory output                 *
*                                                                     *
***********************************************************************
*
      include 'includepar'
      include 'includecom'

      logical error
      integer ldat,ltim
      double precision juldate

      error=.false.

* Open file 'STARTFLIGHT'
*************************

      open(unitpoin,file=path(1)(1:len(1))//'STARTFLIGHT',
     +status='old',err=999)

      call skplin(27,unitpoin)
      read(unitpoin,'(a)',err=998,end=998) compoint(1)(1:40)
      read(unitpoin,*,err=998,end=998) kind(1)
      read(unitpoin,*,err=998,end=998) kindz(1)
      call skplin(1,unitpoin)

C Read the first starting point
*******************************

      read(unitpoin,*,err=998,end=998) ldat,ltim
      read(unitpoin,*,err=998,end=998) xpoint(1)
      read(unitpoin,*,err=998,end=998) ypoint(1)
      read(unitpoin,*,err=998,end=998) zpoint(1)
      call skplin(1,unitpoin)
      nextflight=nint(sngl(juldate(ldat,ltim)-bdate)*86400.)
      if (kindz(1).eq.3) zpoint(1)=zpoint(1)*100.

      numpoint=1

  
C Forbid mixing layer trajectories
**********************************

      if (kind(1).eq.3) then
        write(*,*) '### Mixing layer trajectories not allowed ###'
        write(*,*) '### for FLIGHT calculations. Select a     ###'
        write(*,*) '### different trajectory type!            ###'
        error=.true.
        return
      endif

      return

998   error=.true.
         write(*,*) '#### TRAJECTORY MODEL SUBROUTINE READFLIGHT: '//
     &              '#### '
         write(*,*) '#### FATAL ERROR - FILE "STARTFLIGHT" IS     '//
     &              '#### '
         write(*,*) '#### CORRUPT. PLEASE CHECK YOUR INPUTS FOR   '//
     &              '#### '
         write(*,*) '#### MISTAKES OR GET A NEW "STARTPOINTS"-    '//
     &              '#### '
         write(*,*) '#### FILE ...                                '//
     &              '#### '
      return

999   error=.true.
      write(*,*)  
      write(*,*) ' ###########################################'//
     &           '###### '
      write(*,*) '    TRAJECTORY MODEL SUBROUTINE READFLIGHT:'
      write(*,*)
      write(*,*) ' FATAL ERROR - FILE STARTFLIGHT IS NOT AVAILABLE'
      write(*,*) ' OR YOU ARE NOT PERMITTED FOR ANY ACCESS'
      write(*,*) ' ###########################################'//
     &           '###### '

      return
      end
