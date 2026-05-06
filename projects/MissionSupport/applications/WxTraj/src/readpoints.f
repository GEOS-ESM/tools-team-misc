      subroutine readpoints(error)
***********************************************************************
*                                                                     * 
*             TRAJECTORY MODEL SUBROUTINE READPOINTS                  *
*                                                                     *
***********************************************************************
*                                                                     * 
*             AUTHOR:      G. WOTAWA                                  *
*             DATE:        1994-02-03                                 *
*             LAST UPDATE: 1996-06-21 A. Stohl                        *
*                                                                     * 
* Update: Vertical coordinate can now be given in meters above sea    * 
* level, meters above ground, and in hPa.                             * 
*                                                                     * 
*     10 January 1999  Update to facilitate free formatted input      *
*     (P. Seibert + A. Stohl)                                         *
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

      integer isum
      logical error,old
      integer kindhelp,kindzhelp
      real xhelp,yhelp,zhelp
      character*45 comhelp,line

      error=.false.
*
* OPENING OF FILE 'STARTPOINTS'
*

      open(unitpoin,file=path(1)(1:len(1))//'STARTPOINTS',
     &     status='old',err=999)

C Check the format of the STARTPOINTS file (either in free format,
C or using formatted mask)
C Use of formatted mask is assumed if line 28 contains the word '____'
**********************************************************************

      call skplin(27,unitpoin)
      read (unitpoin,901) line
901   format (a)
      if (index(line,'____') .eq. 0) then
        old = .false.
      else
        old = .true.
      endif
      rewind(unitpoin)

*
* READING OF STARTING/ENDING POINTS OF TRAJECTORIES
*
      call skplin(26,unitpoin)

10    continue
      isum=0
100   read(unitpoin,*,err=998,end=25) xhelp
      if (old) call skplin(2,unitpoin)
      read(unitpoin,*,err=998,end=25) yhelp
      if (old) call skplin(2,unitpoin)
      read(unitpoin,*,err=998,end=25) kindhelp
      if (old) call skplin(2,unitpoin)
      read(unitpoin,*,err=998,end=25) kindzhelp
      if (old) call skplin(2,unitpoin)
      read(unitpoin,*,err=998,end=25) zhelp
      if (old) call skplin(2,unitpoin)
      if (old) then
        read(unitpoin,'(a40)',err=998,end=25) comhelp(1:40)
      else
        read(unitpoin,*,err=998,end=25) comhelp(1:40)
      endif
      if (old) call skplin(1,unitpoin)
      call skplin(1,unitpoin)

      if((xhelp.eq.0.).and.(yhelp.eq.0.).and.
     &   (zhelp.eq.0.).and.(comhelp(1:8).eq.' '))
     &   goto 25


      if ((kindhelp.eq.3).and.(kindzhelp.ne.2)) then
        write(*,*) ' #### FLEXTRA MODEL ERROR! FOR MIXING LAYER   #### '
        write(*,*) ' #### TRAJECTORIES, THE Z COORDINATE MUST BE  #### '
        write(*,*) ' #### GIVEN IN METERS ABOVE GROUND.           #### '
        error=.true.
        return
      endif
  

      isum=isum+1
      if(isum.gt.maxpoint.or.isum*(numbunc+1).gt.maxpoint) then
         error=.true.
         write(*,*) '#### TRAJECTORY MODEL SUBROUTINE READPOINTS: '//
     &              '#### '
         write(*,*) '#### ERROR - TOO MANY STARTING POINTS        '//
     &              '#### '
         write(*,*) '#### REDUCE NUMBER OF STARTING POINTS OR     '//
     &              '#### '
         write(*,*) '#### CHANGE PARAMETERIZATION OF TRAJECTORY   '//
     &              '#### '
         write(*,*) '#### MODEL !!!                               '//
     &              '#### '
         goto 25
      endif
      xpoint(isum)=xhelp
      ypoint(isum)=yhelp
      kind(isum)=kindhelp
      kindz(isum)=kindzhelp
      zpoint(isum)=zhelp
C Conversion from hPa to Pa, if z coordinate is given in pressure units
      if (kindz(isum).eq.3) zpoint(isum)=zpoint(isum)*100. 
      compoint(isum)(1:40)=comhelp(1:40)
      goto 100
1     format(1x,a17,i6,a28)
25    close(unitpoin)
      numpoint=isum

      if ((numpoint*(numbunc+1)*(abs(lentra)/interv+1)).gt.maxtra) then
        write(*,*) ' #### TRAJECTORY MODEL ERROR! MODEL CAN NOT   #### ' 
        write(*,*) ' #### KEEP SO MANY TRAJECTORIES IN MEMORY.    #### '
        write(*,1) ' #### MAXIMUM:   ',maxtra,'                     ####
     + '
        write(*,1) ' #### CURRENTLY: ',numpoint*(numbunc+1)*(abs(lentra)
     +  /interv+1),'                      #### '
        write(*,*) ' #### YOU HAVE 3 POSSIBILITIES:               #### '
        write(*,*) ' #### 1) REDUCE NUMBER OF STARTING POINTS     #### '
        write(*,*) ' ####    (FILE "STARTPOINTS").                #### '
        write(*,*) ' #### 2) REDUCE THE LENGTH OF AN INDIVIDUAL   #### '
        write(*,*) ' ####    TRAJECTORY (FILE "COMMAND").         #### '
        write(*,*) ' #### 3) INCREASE THE TIME INTERVAL BETWEEN   #### '
        write(*,*) ' ####    TRAJECTORY STARTING TIMES (FILE      #### '
        write(*,*) ' ####    "COMMAND").                          #### '
        error=.true.
      endif


      return

998   error=.true.
         write(*,*) '#### TRAJECTORY MODEL SUBROUTINE READPOINTS: '//
     &              '#### '
         write(*,*) '#### FATAL ERROR - FILE "STARTPOINTS" IS     '//
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
      write(*,*) '    TRAJECTORY MODEL SUBROUTINE READPOINTS: '
      write(*,*)
      write(*,*) ' FATAL ERROR - FILE CONTAINING TRAJECTORY ST'//
     &           'ARTING '
      write(*,*) ' AND ENDING POINTS IS NOT AVAILABLE OR YOU A'//
     &           'RE NOT'
      write(*,*) ' PERMITTED FOR ANY ACCESS                   '
      write(*,*) ' ###########################################'//
     &           '###### '

      return
      end
