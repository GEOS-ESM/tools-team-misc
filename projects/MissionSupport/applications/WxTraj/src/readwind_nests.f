      subroutine readwind_nests(indj,n)
C                                i   i
************************************************************************
*                                                                      *
*     This routine reads the wind fields for the nested model domains. *
*     It is similar to subroutine readwind, which reads the mother
*     domain.                                                          *
*                                                                      *
*     Authors: A. Stohl, G. Wotawa                                     *
*                                                                      *
*     30 December 1998                                                 *
*                                                                      *
*     Update:                                                          *
*                          2011-06, implemented reading of grib2 format*
*                                   analog to FLEXPART8.22 routines    *
************************************************************************

      use grib_api
      include 'includepar'
      include 'includecom'

      integer indj,i,j,k,l,n,levdiff2,ifield,iumax,iwmax

!HSO  parameters for grib_api
      integer ifile
      integer iret
      integer igrib
      integer gribVer,parCat,parNum,typSurf,valSurf,discipl
      integer gotGrid
      character*24 gribErrorMsg
      character*20 gribFunction
!HSO  end

* VARIABLES AND ARRAYS NEEDED FOR GRIB DECODING

C dimension of isec2 at least (22+n), where n is the number of parallels or
C meridians in a quasi-regular (reduced) Gaussian or lat/long grid

C dimension of zsec2 at least (10+nn), where nn is the number of vertical
C coordinate parameters

      integer isec0(2),isec1(56),isec2(22+nxmaxn+nymaxn),isec3(2)
      integer isec4(64),inbuff(jpack),ilen,iswap,ierr,lunit,iword
      real zsec2(60+2*nuvzmax),zsec3(2),zsec4(jpunp)
      real xaux,yaux,xaux0,yaux0
      real*8 xauxin,yauxin

      character*1 yoper
      logical error

      data yoper/'D'/
!HSO  grib api error messages
      data gribErrorMsg/'Error reading grib file'/
      data gribFunction/'readwind_nests'/

      do 100 l=1,numbnests
        levdiff2=nlev_ec-nwz+1
        iumax=0
        iwmax=0

        ifile=0
        igrib=0
        iret=0
*
* OPENING OF DATA FILE (GRIB CODE)
*
5       call grib_open_file(ifile,path(numpath+2*(l-1)+1)
     +   (1:len(numpath+2*(l-1)+1))//trim(wfnamen(l,indj)),'r')
        if (iret.ne.GRIB_SUCCESS) then
          goto 888   ! ERROR DETECTED
        endif

        gotGrid=0
        ifield=0
C   loop fields
10      ifield=ifield+1
*
* GET NEXT FIELDS
*
        call grib_new_from_file(ifile,igrib,iret)
        if (iret.eq.GRIB_END_OF_FILE)  then
          goto 50    ! EOF DETECTED
        elseif (iret.ne.GRIB_SUCCESS) then
          goto 888   ! ERROR DETECTED
        endif

!     first see if we read GRIB1 or GRIB2
        call grib_get_int(igrib,'editionNumber',gribVer,iret)
        call grib_check(iret,gribFunction,gribErrorMsg)

        if (gribVer.eq.1) then ! GRIB Edition 1
c     print*,'GRiB Edition 1'
c     read the grib2 identifiers
          call grib_get_int(igrib,'indicatorOfParameter',isec1(6),
     +     iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
          call grib_get_int(igrib,'level',isec1(8),iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
c     change code for etadot to code for omega
          if (isec1(6).eq.77) then
            isec1(6)=135
          endif
        else
c     print*,'GRiB Edition 2'
c     read the grib2 identifiers
          call grib_get_int(igrib,'discipline',discipl,iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
          call grib_get_int(igrib,'parameterCategory',parCat,iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
          call grib_get_int(igrib,'parameterNumber',parNum,iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
          call grib_get_int(igrib,'typeOfFirstFixedSurface',typSurf,
     +     iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
          call grib_get_int(igrib,'level',valSurf,iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
!     convert to grib1 identifiers
          isec1(6)=-1
          isec1(7)=-1
          isec1(8)=-1
          isec1(8)=valSurf     ! level
          if((parCat.eq.0).and.(parNum.eq.0).and.(typSurf.eq.105))then ! T
            isec1(6)=130         ! indicatorOfParameter
          elseif((parCat.eq.2).and.(parNum.eq.2).and.(typSurf.eq.105))
     +     then ! U
            isec1(6)=131         ! indicatorOfParameter
          elseif((parCat.eq.2).and.(parNum.eq.3).and.(typSurf.eq.105))
     +     then ! V
            isec1(6)=132         ! indicatorOfParameter
          elseif((parCat.eq.1).and.(parNum.eq.0).and.(typSurf.eq.105))
     +     then ! Q
            isec1(6)=133         ! indicatorOfParameter
          elseif((parCat.eq.3).and.(parNum.eq.0).and.(typSurf.eq.1))then !SP
            isec1(6)=134         ! indicatorOfParameter
          elseif((parCat.eq.2).and.(parNum.eq.32)) then ! W, actually eta dot
            isec1(6)=135         ! indicatorOfParameter
          elseif((parCat.eq.3).and.(parNum.eq.0).and.(typSurf.eq.101))
     +     then !SLP
            isec1(6)=151         ! indicatorOfParameter
          elseif((parCat.eq.2).and.(parNum.eq.2).and.(typSurf.eq.103))
     +     then ! 10U
            isec1(6)=165         ! indicatorOfParameter
          elseif((parCat.eq.2).and.(parNum.eq.3).and.(typSurf.eq.103))
     +     then ! 10V
            isec1(6)=166         ! indicatorOfParameter
          elseif((parCat.eq.0).and.(parNum.eq.0).and.(typSurf.eq.103))
     +     then ! 2T
            isec1(6)=167         ! indicatorOfParameter
          elseif((parCat.eq.0).and.(parNum.eq.6).and.(typSurf.eq.103))
     +     then ! 2D
            isec1(6)=168         ! indicatorOfParameter
          elseif((parCat.eq.1).and.(parNum.eq.11).and.(typSurf.eq.1))
     +     then ! SD
            isec1(6)=141         ! indicatorOfParameter
          elseif((parCat.eq.6).and.(parNum.eq.1)) then ! CC
            isec1(6)=164         ! indicatorOfParameter
          elseif((parCat.eq.1).and.(parNum.eq.9)) then ! LSP
            isec1(6)=142         ! indicatorOfParameter
          elseif((parCat.eq.1).and.(parNum.eq.10)) then ! CP
            isec1(6)=143         ! indicatorOfParameter
          elseif((parCat.eq.0).and.(parNum.eq.11).and.(typSurf.eq.1))
     +     then ! SHF
            isec1(6)=146         ! indicatorOfParameter
          elseif((parCat.eq.4).and.(parNum.eq.9).and.(typSurf.eq.1))
     +     then ! SR
            isec1(6)=176         ! indicatorOfParameter
          elseif((parCat.eq.2).and.(parNum.eq.17)) then ! EWSS
            isec1(6)=180         ! indicatorOfParameter
          elseif((parCat.eq.2).and.(parNum.eq.18)) then ! NSSS
            isec1(6)=181         ! indicatorOfParameter
          elseif((parCat.eq.3).and.(parNum.eq.4)) then ! ORO
            isec1(6)=129         ! indicatorOfParameter
          elseif((parCat.eq.3).and.(parNum.eq.7)) then ! SDO
            isec1(6)=160         ! indicatorOfParameter
          elseif((discipl.eq.2).and.(parCat.eq.0).and.(parNum.eq.0).and.
     +      (typSurf.eq.1)) then ! LSM
            isec1(6)=172         ! indicatorOfParameter
          else
            print*,'***ERROR: undefined GRiB2 message found!',discipl,
     +        parCat,parNum,typSurf
          endif
        endif ! GRIB version

!HSO  get the size and data of the values array
        if (isec1(6).ne.-1) then
          call grib_get_real4_array(igrib,'values',zsec4,iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
        endif

!HSO  get the required fields from section 2 in a gribex compatible manner
        if(ifield.eq.1) then
        call grib_get_int(igrib,'numberOfPointsAlongAParallel',
     +     isec2(2),iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
        call grib_get_int(igrib,'numberOfPointsAlongAMeridian',
     1     isec2(3),iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
        call grib_get_int(igrib,'numberOfVerticalCoordinateValues',
     +     isec2(12))
        call grib_check(iret,gribFunction,gribErrorMsg)
* CHECK GRID SPECIFICATIONS
        if(isec2(2).ne.nxn(l)) stop
     +    'READWIND: NX NOT CONSISTENT FOR A NESTING LEVEL'
        if(isec2(3).ne.nyn(l)) stop
     +    'READWIND: NY NOT CONSISTENT FOR A NESTING LEVEL'
        if(isec2(12)/2-1.ne.nlev_ec) stop 'READWIND: VERTICAL DISCRET
     +IZATION NOT CONSISTENT FOR A NESTING LEVEL'
        endif ! ifield
!HSO  get the second part of the grid dimensions only from GRiB1 messages
        if ((gribVer.eq.1).and.(gotGrid.eq.0)) then
          call grib_get_real8(igrib,'longitudeOfFirstGridPointInDegrees'
     +     ,xauxin,iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
          call grib_get_real8(igrib,'latitudeOfLastGridPointInDegrees',
     +     yauxin,iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
          xaux=xauxin
          yaux=yauxin
          xaux0=xlon0n(l)
          yaux0=ylat0n(l)
          if(xaux.lt.0.) xaux=xaux+360.
          if(yaux.lt.0.) yaux=yaux+360.
          if(xaux0.lt.0.) xaux0=xaux0+360.
          if(yaux0.lt.0.) yaux0=yaux0+360.
          if(xaux.ne.xaux0)
     +     stop 'READWIND: LOWER LEFT LONGITUDE NOT CONSISTENT FOR A NES
     +TING LEVEL'
          if(yaux.ne.yaux0)
     +      stop 'READWIND: LOWER LEFT LATITUDE NOT CONSISTENT FOR A NEST
     +ING LEVEL'
          gotGrid=1
        endif

          do 20 j=0,nyn(l)-1
            do 20 i=0,nxn(l)-1
              k=isec1(8)
              if(isec1(6).eq.130) ttn(i,j,nlev_ec-k+1,n,l)=    !! TEMPERATURE
     &        zsec4(nxn(l)*(nyn(l)-j-1)+i+1)
              if(isec1(6).eq.131) uun(i,j,nlev_ec-k+1,n,l)=    !! U VELOCITY
     &        zsec4(nxn(l)*(nyn(l)-j-1)+i+1)
              if(isec1(6).eq.132) vvn(i,j,nlev_ec-k+1,n,l)=    !! V VELOCITY
     &        zsec4(nxn(l)*(nyn(l)-j-1)+i+1)
              if(isec1(6).eq.133) then
                qqn(i,j,nlev_ec-k+1,n,l)=                      !! HUMIDITY
     &          zsec4(nxn(l)*(nyn(l)-j-1)+i+1)
                if (qqn(i,j,nlev_ec-k+1,n,l).lt.0.)
     &          qqn(i,j,nlev_ec-k+1,n,l)=0.
c             this is necessary because the gridded data may contain
c             spurious negative values
              endif
              if(isec1(6).eq.134) psn(i,j,1,n,l)=              !! SURF. PRESS.
     &        zsec4(nxn(l)*(nyn(l)-j-1)+i+1)
              if(isec1(6).eq.135) wwn(i,j,nlev_ec-k+1,n,l)=    !! W VELOCITY
     &        zsec4(nxn(l)*(nyn(l)-j-1)+i+1)
              if(isec1(6).eq.129) oron(i,j,l)=                 !! ECMWF OROGRAPHY
     &        zsec4(nxn(l)*(nyn(l)-j-1)+i+1)/ga
              if(isec1(6).eq.131) iumax=max(iumax,nlev_ec-k+1)
              if(isec1(6).eq.135) iwmax=max(iwmax,nlev_ec-k+1)
20            continue

        call grib_release(igrib)
        goto 10                      !! READ NEXT LEVEL OR PARAMETER
*
* CLOSING OF INPUT DATA FILE
*
50      call grib_close_file(ifile)
*     error message if no fields found with correct first longitude in it
        if (gotGrid.eq.0) then
          print*,'***ERROR: input file needs to contain GRiB1 formatted'
     +     //'messages'
          stop
        endif

        if(levdiff2.eq.0) then
          iwmax=nlev_ec+1
          do 60 j=0,nyn(l)-1
            do 60 i=0,nxn(l)-1
60            wwn(i,j,nlev_ec+1,n,l)=0.
        endif


        if(iumax.ne.nuvz) stop
     +  'READWIND: NUVZ NOT CONSISTENT FOR A NESTING LEVEL'
        if(iwmax.ne.nwz) stop
     +  'READWIND: NWZ NOT CONSISTENT FOR A NESTING LEVEL'



C Calculate potential temperature and potential vorticity on whole grid
***********************************************************************
        call calcpv_nests(l,n)

100   continue  ! END OF NESTS LOOP

      return
888   write(*,*) ' #### TRAJECTORY MODEL ERROR! WINDFIELD       #### '
      write(*,*) ' #### ',wfname(indj),'                    #### '
      write(*,*) ' #### IS NOT GRIB FORMAT !!!                  #### '
      stop 'Execution terminated'


999   write(*,*)
      write(*,*) ' ###########################################'//
     &           '###### '
      write(*,*) '       TRAJECTORY MODEL SUBROUTINE GRIDCHECK:'
      write(*,*) ' CAN NOT OPEN INPUT DATA FILE '//wfnamen(l,indj)
      write(*,*) ' FOR NESTING LEVEL ',l
      write(*,*) ' ###########################################'//
     &           '###### '
      error=.true.

      end
