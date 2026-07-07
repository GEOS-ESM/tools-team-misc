      subroutine gridcheck_nests(error)
C                                  o
********************************************************************************
*                                                                              *
*     This routine checks the grid specification for the nested model domains. *
*     It is similar to subroutine gridcheck, which checks the mother domain.   *
*                                                                              *
*     Authors: A. Stohl, G. Wotawa                                             *
*                                                                              *
*     30 December 1998                                                         *
*                                                                      *
*     Updated:                                                         *
*                          2011-06, implemented reading of grib2 format*
*                                   analog to FLEXPART 8.22 routines   *
************************************************************************
      use grib_api

      include 'includepar'
      include 'includecom'

!HSO  parameters for grib_api
      integer ifile
      integer iret
      integer igrib
      integer gribVer,parCat,parNum,typSurf,valSurf,discipl
      integer gotGrib
      character*24 gribErrorMsg
      character*20 gribFunction
!HSO  end

      integer i,j,k,l,ifn,ifield,iumax,iwmax,numskip,nlev_ecn
      integer nuvzn,nwzn
      real akmn(nwzmax),bkmn(nwzmax),akzn(nuvzmax),bkzn(nuvzmax)
      real uvheightn(nuvzmax),wheightn(nwzmax)
      real xaux1,xaux2,yaux1,yaux2
      real*8 xaux1in,xaux2in,yaux1in,yaux2in
      logical error,oronew

* VARIABLES AND ARRAYS NEEDED FOR GRIB DECODING

C dimension of isec2 at least (22+n), where n is the number of parallels or
C meridians in a quasi-regular (reduced) Gaussian or lat/long grid

C dimension of zsec2 at least (10+nn), where nn is the number of vertical
C coordinate parameters

      integer isec0(2),isec1(56),isec2(22+nxmaxn+nymaxn),isec3(2)
      integer isec4(64),inbuff(jpack),ilen,iswap,ierr,iword,lunit
      real zsec2(60+2*nuvzmax),zsec3(2),zsec4(jpunp)
      character*1 yoper
!HSO  grib api error messages
      data gribErrorMsg/'Error reading grib file'/
      data gribFunction/'gridcheck_nests'/

      data yoper/'D'/

      error=.false.
      xresoln(0)=1.       ! resolution enhancement for mother grid
      yresoln(0)=1.       ! resolution enhancement for mother grid

C Loop about all nesting levels
*******************************

      do 300 l=1,numbnests
        oronew=.false.

        iumax=0
        iwmax=0
*
        if(ideltas.gt.0) then
          ifn=1
        else
          ifn=numbwf
        endif
*
* OPENING OF DATA FILE (GRIB CODE)
*
        ifile=0
        igrib=0
        iret=0

5       call grib_open_file(ifile,path(numpath+2*(l-1)+1)
     +   (1:len(numpath+2*(l-1)+1))//trim(wfnamen(l,ifn)),'r',iret)
        if (iret.ne.GRIB_SUCCESS) then
          goto 999   ! ERROR DETECTED
        endif

        gotGrib=0
        ifield=0
10        ifield=ifield+1
*
* GET NEXT FIELDS
*
        call grib_new_from_file(ifile,igrib,iret)
        if (iret.eq.GRIB_END_OF_FILE)  then
          goto 30    ! EOF DETECTED
        elseif (iret.ne.GRIB_SUCCESS) then
          goto 999   ! ERROR DETECTED
        endif

!     first see if we read GRIB1 or GRIB2
        call grib_get_int(igrib,'editionNumber',gribVer,iret)
        call grib_check(iret,gribFunction,gribErrorMsg)

        if (gribVer.eq.1) then ! GRIB Edition 1
c     print*,'GRiB Edition 1'
c     read the grib2 identifiers
          call grib_get_int(igrib,'indicatorOfParameter',isec1(6),iret)
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
        endif
!     get the size and data of the values array
        if (isec1(6).ne.-1) then
          call grib_get_real4_array(igrib,'values',zsec4,iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
        endif

!HSO  get the required fields from section 2 in a gribex compatible manner
        if (ifield.eq.1) then
          call grib_get_int(igrib,'numberOfPointsAlongAParallel',
     >     isec2(2),iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
          call grib_get_int(igrib,'numberOfPointsAlongAMeridian',
     >     isec2(3),iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
          call grib_get_int(igrib,'numberOfVerticalCoordinateValues',
     >     isec2(12),iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
!HSO    get the size and data of the vertical coordinate array
          call grib_get_real4_array(igrib,'pv',zsec2,iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
          nxn(l)=isec2(2)
          nyn(l)=isec2(3)
          nlev_ecn=isec2(12)/2-1
        endif ! ifield

!HSO  get the second part of the grid dimensions only from GRiB1 messages
        if ((gribVer.eq.1).and.(gotGrib.eq.0)) then
          call grib_get_real8(igrib,'longitudeOfFirstGridPointInDegrees'
     +       ,xaux1in,iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
          call grib_get_real8(igrib,'longitudeOfLastGridPointInDegrees',
     +       xaux2in,iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
          call grib_get_real8(igrib,'latitudeOfLastGridPointInDegrees',
     +       yaux1in,iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
          call grib_get_real8(igrib,'latitudeOfFirstGridPointInDegrees',
     +       yaux2in,iret)
          call grib_check(iret,gribFunction,gribErrorMsg)
          xaux1=xaux1in
          xaux2=xaux2in
          yaux1=yaux1in
          yaux2=yaux2in
          if(xaux1.gt.180) xaux1=xaux1-360.0
          if(xaux2.gt.180) xaux2=xaux2-360.0
          if(xaux1.lt.-180) xaux1=xaux1+360.0
          if(xaux2.lt.-180) xaux2=xaux2+360.0
          if (xaux2.lt.xaux1) xaux2=xaux2+360.
          xlon0n(l)=xaux1
          ylat0n(l)=yaux1
          dxn(l)=(xaux2-xaux1)/float(nxn(l)-1)
          dyn(l)=(yaux2-yaux1)/float(nyn(l)-1)
          gotGrib=1
        endif ! ifield.eq.1


        if(isec1(6).eq.129) oronew=.true.
        k=isec1(8)
        if(isec1(6).eq.131) iumax=max(iumax,nlev_ecn-k+1)
        if(isec1(6).eq.135) iwmax=max(iwmax,nlev_ecn-k+1)

        if(isec1(6).eq.129) then
          do 20 j=0,nyn(l)-1
            do 21 i=0,nxn(l)-1
21            oron(i,j,l)=zsec4(nxn(l)*(nyn(l)-j-1)+i+1)/ga
20          continue
        endif

        call grib_release(igrib)
        goto 10                  ! READ NEXT LEVEL OR PARAMETER
*
* CLOSING OF INPUT DATA FILE
*
30      call grib_close_file(ifile)

*     error message if no fields found with correct first longitude in it
        if (gotGrib.eq.0) then
          print*,'***ERROR: input file needs to contain GRiB1 formatted'
     +  //'messages'
          stop
        endif

        nuvzn=iumax
        nwzn=iwmax
        if(nuvzn.eq.nlev_ecn) nwzn=nlev_ecn+1

        if (nxn(l).gt.nxmaxn) then
      write(*,*) 'FLEXTRA error: Too many grid points in x direction.'
      write(*,*) 'Reduce resolution of wind fields (file GRIDSPEC)'
      write(*,*) 'for nesting level ',l

          error=.true.
          return
        endif

        if (nyn(l).gt.nymaxn) then
      write(*,*) 'FLEXTRA error: Too many grid points in y direction.'
      write(*,*) 'Reduce resolution of wind fields (file GRIDSPEC)'
      write(*,*) 'for nesting level ',l
          error=.true.
          return
        endif

        if ((nuvzn.gt.nuvzmax).or.(nwzn.gt.nwzmax)) then
      write(*,*) 'FLEXTRA error: Nested wind fields have too many'//
     +'vertical levels.'
      write(*,*) 'Problem was encountered for nesting level ',l
          error=.true.
          return
        endif


C Output of grid info
*********************

      write(*,'(a,i2)') 'Nested domain #: ',l
      write(*,'(a,f10.2,a1,f10.2,a,f10.2)') '  Longitude range: ',
     +xlon0n(l),' to ',xlon0n(l)+(nxn(l)-1)*dxn(l),
     +'   Grid distance: ',dxn(l)
      write(*,'(a,f10.2,a1,f10.2,a,f10.2)') '  Latitude range:  ',
     +ylat0n(l),' to ',ylat0n(l)+(nyn(l)-1)*dyn(l),
     +'   Grid distance: ',dyn(l)
      write(*,*)


C Determine, how much the resolutions in the nests are enhanced as
C compared to the mother grid
******************************************************************

        xresoln(l)=dx/dxn(l)
        yresoln(l)=dy/dyn(l)

C Determine the mother grid coordinates of the corner points of the
C nested grids
C Convert first to geographical coordinates, then to grid coordinates
*********************************************************************

        xaux1=xlon0n(l)
        xaux2=xlon0n(l)+float(nxn(l)-1)*dxn(l)
        yaux1=ylat0n(l)
        yaux2=ylat0n(l)+float(nyn(l)-1)*dyn(l)

        xln(l)=(xaux1-xlon0)/dx
        xrn(l)=(xaux2-xlon0)/dx
        yln(l)=(yaux1-ylat0)/dy
        yrn(l)=(yaux2-ylat0)/dy


* CALCULATE VERTICAL DISCRETIZATION OF ECMWF MODEL
* PARAMETER akm,bkm DESCRIBE THE HYBRID "ETA" COORDINATE SYSTEM
* wheight(i) IS THE HEIGHT OF THE i-th MODEL HALF LEVEL (=INTERFACE BETWEEN
* 2 MODEL LEVELS) IN THE "ETA" SYSTEM

        numskip=nlev_ecn-nuvzn ! number of ECMWF model layers not used
                               ! by trajectory model
        do 40 i=1,nwzn
          j=numskip+i
          k=nlev_ecn+1+numskip+i
          akmn(nwzn-i+1)=zsec2(j)
          bkmn(nwzn-i+1)=zsec2(k)
40        wheightn(nwzn-i+1)=akmn(nwzn-i+1)/p0+bkmn(nwzn-i+1)

* CALCULATION OF uvheight, akz, bkz
* akz,bkz ARE THE DISCRETIZATION PARAMETERS FOR THE MODEL LEVELS
* uvheight(i) IS THE HEIGHT OF THE i-th MODEL LEVEL IN THE "ETA" SYSTEM

        do 45 i=1,nuvzn
          uvheightn(i)=0.5*(wheightn(i+1)+wheightn(i))
          akzn(i)=0.5*(akmn(i+1)+akmn(i))
45        bkzn(i)=0.5*(bkmn(i+1)+bkmn(i))


C If vertical coordinates decrease with increasing altitude, multiply by -1.
C This means that also the vertical velocities have to be multiplied by -1.
****************************************************************************

        if (uvheightn(1).lt.uvheightn(nuvzn)) then
          zdirect=1.
        else
          zdirect=-1.
          do 55 i=1,nuvz
55          uvheightn(i)=zdirect*uvheightn(i)
          do 65 i=1,nwz
65          wheightn(i)=zdirect*wheightn(i)
        endif


C Check, whether the heights of the model levels of the nested
C wind fields are consistent with those of the mother domain.
C If not, terminate model run.
**************************************************************

        do 75 i=1,nuvz
          if ((akzn(i).ne.akz(i)).or.(bkzn(i).ne.bkz(i)).or.
     +    (uvheightn(i).ne.uvheight(i))) then
      write(*,*) 'FLEXTRA error: The wind fields of nesting level',l
      write(*,*) 'are not consistent with the mother domain:'
      write(*,*) 'Differences in vertical levels detected.'
          error=.true.
          return
          endif
75        continue

        do 85 i=1,nwz
          if ((akmn(i).ne.akm(i)).or.(bkmn(i).ne.bkm(i)).or.
     +    (wheightn(i).ne.wheight(i))) then
      write(*,*) 'FLEXTRA error: The wind fields of nesting level',l
      write(*,*) 'are not consistent with the mother domain:'
      write(*,*) 'Differences in vertical levels detected.'
          error=.true.
          return
          endif
85        continue

        if (.not.oronew) then
          write(*,*) 'FLEXTRA error: Orography of nested domains'
          write(*,*) 'is missing in the wind field files.'
        endif

300     continue
      return


999   write(*,*)
      write(*,*) ' ###########################################'//
     &           '###### '
      write(*,*) '       TRAJECTORY MODEL SUBROUTINE GRIDCHECK:'
      write(*,*) ' CAN NOT OPEN INPUT DATA FILE '//wfnamen(l,ifn)
      write(*,*) ' FOR NESTING LEVEL ',k
      write(*,*) ' ###########################################'//
     &           '###### '
      error=.true.

      return
      end
