      subroutine gridcheck(oronew,error)
************************************************************************
*                                                                      *
*             TRAJECTORY MODEL SUBROUTINE GRIDCHECK                    *
*                                                                      *
************************************************************************
*                                                                      *
*             AUTHOR:      G. WOTAWA                                   *
*             DATE:        1997-08-06                                  *
*                                                                      *
*             Update:      1998-12, global fields allowed, A. Stohl    *
*                          2011-06, implemented reading of grib2 format*
*                                   analog to FLEXPART 8.22 routines   *
*                                                                      *
************************************************************************
*                                                                     *
* DESCRIPTION:                                                        *
*                                                                     *
* THIS SUBROUTINE DETERMINES THE GRID SPECIFICATIONS (LOWER LEFT      *
* LONGITUDE, LOWER LEFT LATITUDE, NUMBER OF GRID POINTS, GRID DIST-   *
* ANCE AND VERTICAL DISCRETIZATION OF THE ECMWF MODEL) FROM THE       *
* GRIB HEADER OF THE FIRST INPUT FILE. THE CONSISTANCY (NO CHANGES    *
* WITHIN ONE FLEXTRA RUN) IS CHECKED IN THE ROUTINE "READWIND" AT ANY *
* CALL.                                                               *
*                                                                     *
* OUTPUT       error .true.   - can not read grid specifications      *
*              error .false.  - normal                                *
*              oronew .true.  - Terrain heights given in grib files   *
*              oronew .false. - Terrain heights not specified in the  *
*                               grib files (old file standard)        *
*                                                                     *
* XLON0                geographical longitude of lower left gridpoint *
* XLAT0                geographical latitude of lower left gridpoint  *
* NX                   number of grid points x-direction              *
* NY                   number of grid points y-direction              *
* DX                   grid distance x-direction                      *
* DY                   grid distance y-direction                      *
* NUVZ                 number of grid points for horizontal wind      *
*                      components in z direction                      *
* NWZ                  number of grid points for vertical wind        *
* sizesouth, sizenorth give the map scale (i.e. number of virtual grid*
*                      points of the polar stereographic grid):       *
*                      used to check the CFL criterion                *
*                      component in z direction                       *
* UVHEIGHT(1)-         heights of gridpoints where u and v are        *
* UVHEIGHT(NUVZ)       given                                          *
* WHEIGHT(1)-          heights of gridpoints where w is given         *
* WHEIGHT(NWZ)                                                        *
*                                                                     *
***********************************************************************
*
      use grib_api

      include 'includepar'
      include 'includecom'

      integer i,ifn,ifield,j,k,iumax,iwmax,numskip
      real sizesouth,sizenorth,xauxa
      logical error,oronew

!HSO  parameters for grib_api
      integer ifile
      integer iret
      integer igrib
      integer gotGrid
      real*4  xaux1,xaux2,yaux1,yaux2
      real*8  xaux1in,xaux2in,yaux1in,yaux2in
      integer gribVer,parCat,parNum,typSurf,valSurf,discipl
      character*24 gribErrorMsg
      character*20 gribFunction
!HSO  end

* VARIABLES AND ARRAYS NEEDED FOR GRIB DECODING

C dimension of isec2 at least (22+n), where n is the number of parallels or
C meridians in a quasi-regular (reduced) Gaussian or lat/long grid

C dimension of zsec2 at least (10+nn), where nn is the number of vertical
C coordinate parameters

      integer isec0(2),isec1(56),isec2(22+nxmax+nymax),isec3(2)
      integer isec4(64),inbuff(jpack),ilen,iswap,ierr,iword,lunit
      real zsec2(91+2*nuvzmax),zsec3(2),zsec4(jpunp)
      character*1 yoper,opt
      data yoper/'D'/

      error=.false.
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
5     call grib_open_file(ifile,path(3)(1:len(3))
     >//trim(wfname(ifn)),'r',iret)
      if (iret.ne.GRIB_SUCCESS) then
        goto 999   ! ERROR DETECTED
      endif
!     turn on support for multi fields messages
!     call grib_multi_support_on()

      gotGrid=0
      ifield=0
10    ifield=ifield+1

*
* GET NEXT FIELDS
*
      call grib_new_from_file(ifile,igrib,iret)
      if (iret.eq.GRIB_END_OF_FILE )  then
        goto 30    ! EOF DETECTED
      elseif (iret.ne.GRIB_SUCCESS) then
        goto 999   ! ERROR DETECTED
      endif

!     first see if we read GRIB1 or GRIB2
      call grib_get_int(igrib,'editionNumber',gribVer,iret)
      call grib_check(iret,gribFunction,gribErrorMsg)

C     GRIB 1
C *******************************************************************
      if (gribVer.eq.1) then
C        print*,'GRiB Edition 1'
c     read the grib2 identifiers
        call grib_get_int(igrib,'indicatorOfParameter',isec1(6),iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
        call grib_get_int(igrib,'level',isec1(8),iret)
        call grib_check(iret,gribFunction,gribErrorMsg)

c     change code for etadot to code for omega
        if (isec1(6).eq.77) then
          isec1(6)=135
        endif
C     GRIB 2
C *******************************************************************
      else
C        print*,'GRiB Edition 2'
c     read the grib2 identifiers
        call grib_get_int(igrib,'discipline',discipl,iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
        call grib_get_int(igrib,'parameterCategory',parCat,iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
        call grib_get_int(igrib,'parameterNumber',parNum,iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
        call grib_get_int(igrib,'typeOfFirstFixedSurface',typSurf,iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
        call grib_get_int(igrib,'level',valSurf,iret)
        call grib_check(iret,gribFunction,gribErrorMsg)

!     convert to grib1 identifiers
        isec1(6)=-1
        isec1(7)=-1
        isec1(8)=-1
        isec1(8)=valSurf     ! level

        if ((parCat.eq.0).and.(parNum.eq.0).and.(typSurf.eq.105))then ! T
          isec1(6)=130         ! indicatorOfParameter
        elseif((parCat.eq.2).and.(parNum.eq.2).and.(typSurf.eq.105))then ! U
          isec1(6)=131         ! indicatorOfParameter
        elseif((parCat.eq.2).and.(parNum.eq.3).and.(typSurf.eq.105))then ! V
          isec1(6)=132         ! indicatorOfParameter
        elseif((parCat.eq.1).and.(parNum.eq.0).and.(typSurf.eq.105))then ! Q
          isec1(6)=133         ! indicatorOfParameter
        elseif((parCat.eq.3).and.(parNum.eq.0).and.(typSurf.eq.1))then !SP
          isec1(6)=134         ! indicatorOfParameter
        elseif((parCat.eq.2).and.(parNum.eq.32)) then ! W, actually eta dot
          isec1(6)=135         ! indicatorOfParameter
        elseif((parCat.eq.3).and.(parNum.eq.0).and.(typSurf.eq.101))then !SLP
          isec1(6)=151         ! indicatorOfParameter
        elseif((parCat.eq.2).and.(parNum.eq.2).and.(typSurf.eq.103))then ! 10U
          isec1(6)=165         ! indicatorOfParameter
        elseif((parCat.eq.2).and.(parNum.eq.3).and.(typSurf.eq.103))then ! 10V
          isec1(6)=166         ! indicatorOfParameter
        elseif((parCat.eq.0).and.(parNum.eq.0).and.(typSurf.eq.103))then ! 2T
          isec1(6)=167         ! indicatorOfParameter
        elseif((parCat.eq.0).and.(parNum.eq.6).and.(typSurf.eq.103))then ! 2D
          isec1(6)=168         ! indicatorOfParameter
        elseif((parCat.eq.1).and.(parNum.eq.11).and.(typSurf.eq.1))then ! SD
          isec1(6)=141         ! indicatorOfParameter
        elseif((parCat.eq.6).and.(parNum.eq.1)) then ! CC
          isec1(6)=164         ! indicatorOfParameter
        elseif((parCat.eq.1).and.(parNum.eq.9)) then ! LSP
          isec1(6)=142         ! indicatorOfParameter
        elseif((parCat.eq.1).and.(parNum.eq.10)) then ! CP
          isec1(6)=143         ! indicatorOfParameter
        elseif((parCat.eq.0).and.(parNum.eq.11).and.(typSurf.eq.1))then ! SHF
          isec1(6)=146         ! indicatorOfParameter
        elseif((parCat.eq.4).and.(parNum.eq.9).and.(typSurf.eq.1))then ! SR
          isec1(6)=176         ! indicatorOfParameter
        elseif((parCat.eq.2).and.(parNum.eq.17))then ! EWSS
          isec1(6)=180         ! indicatorOfParameter
        elseif((parCat.eq.2).and.(parNum.eq.18))then ! NSSS
          isec1(6)=181         ! indicatorOfParameter
        elseif((parCat.eq.3).and.(parNum.eq.4))then ! ORO
          isec1(6)=129         ! indicatorOfParameter
        elseif((parCat.eq.3).and.(parNum.eq.7))then ! SDO
          isec1(6)=160         ! indicatorOfParameter
        elseif((discipl.eq.2).and.(parCat.eq.0).and.(parNum.eq.0).and.
     +     (typSurf.eq.1)) then ! LSM
          isec1(6)=172         ! indicatorOfParameter
        else
          print*,'***ERROR: undefined GRiB2 message found!',discipl,
     +     parCat,parNum,typSurf
        endif


      endif

!     get the size and data of the values array
      if (isec1(6).ne.-1) then
        call grib_get_real4_array(igrib,'values',zsec4,iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
      endif

      if(ifield.eq.1) then
!HSO  get the required fields from section 2 in a gribex compatible manner
        call grib_get_int(igrib,'numberOfPointsAlongAParallel',
     >       isec2(2),iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
        call grib_get_int(igrib,'numberOfPointsAlongAMeridian',
     >       isec2(3),iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
        call grib_get_real8(igrib,'longitudeOfFirstGridPointInDegrees',
     >       xaux1in,iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
        call grib_get_int(igrib,'numberOfVerticalCoordinateValues',
     >       isec2(12),iret)
        call grib_check(iret,gribFunction,gribErrorMsg)

!       get the size and data of the vertical coordinate array
        call grib_get_real4_array(igrib,'pv',zsec2,iret)
        call grib_check(iret,gribFunction,gribErrorMsg)

        nxfield=isec2(2)
        ny=isec2(3)
        nlev_ec=isec2(12)/2-1
      endif

!HSO  get the second part of the grid dimensions only from GRiB1 messages
      if ((gribVer.eq.1).and.(gotGrid.eq.0)) then
        call grib_get_real8(igrib,'longitudeOfLastGridPointInDegrees',
     >       xaux2in,iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
        call grib_get_real8(igrib,'latitudeOfLastGridPointInDegrees',
     >       yaux1in,iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
        call grib_get_real8(igrib,'latitudeOfFirstGridPointInDegrees',
     >       yaux2in,iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
        xaux1=xaux1in
        xaux2=xaux2in
        yaux1=yaux1in
        yaux2=yaux2in
        if (xaux1.gt.180.) xaux1=xaux1-360.0
        if (xaux2.gt.180.) xaux2=xaux2-360.0
        if (xaux1.lt.-180.) xaux1=xaux1+360.0
        if (xaux2.lt.-180.) xaux2=xaux2+360.0
        if (xaux2.lt.xaux1) xaux2=xaux2+360.0
        xlon0=xaux1
        ylat0=yaux1
        dx=(xaux2-xaux1)/float(nxfield-1)
        dy=(yaux2-yaux1)/float(ny-1)
        gotGrid=1

C Check whether fields are global
C If they contain the poles, specify polar stereographic map
C projections using the stlmbr- and stcm2p-calls
************************************************************

        if (xauxa.lt.0.001) then
          nx=nxfield+1                 ! field is cyclic
          xglobal=.true.
        else
          nx=nxfield
          xglobal=.false.
        endif
        if (xlon0.gt.180.) xlon0=xlon0-360.
        xauxa=abs(yaux1+90.)
        if (xglobal.and.xauxa.lt.0.001) then
          sglobal=.true.               ! field contains south pole
C Enhance the map scale by factor 3 (*2=6) compared to north-south
C map scale
          sizesouth=6.*(switchsouth+90.)/dy
          call stlmbr(southpolemap,-90.,0.)
          call stcm2p(southpolemap,0.,0.,switchsouth,0.,sizesouth,
     +    sizesouth,switchsouth,180.)
          switchsouthg=(switchsouth-ylat0)/dy
        else
          sglobal=.false.
          switchsouthg=999999.
        endif
        xauxa=abs(yaux2-90.)
        if (xglobal.and.xauxa.lt.0.001) then
          nglobal=.true.               ! field contains north pole
C Enhance the map scale by factor 3 (*2=6) compared to north-south
C map scale
          sizenorth=6.*(90.-switchnorth)/dy
          call stlmbr(northpolemap,90.,0.)
          call stcm2p(northpolemap,0.,0.,switchnorth,0.,sizenorth,
     +    sizenorth,switchnorth,180.)
          switchnorthg=(switchnorth-ylat0)/dy
        else
          nglobal=.false.
          switchnorthg=999999.
        endif
      endif ! gotGrid

      k=isec1(8)
      if(isec1(6).eq.131) iumax=max(iumax,nlev_ec-k+1)
      if(isec1(6).eq.135) iwmax=max(iwmax,nlev_ec-k+1)

      if(isec1(6).eq.129) then
        oronew=.true.
        do 20 j=0,ny-1
          do 21 i=0,nxfield-1
21          oro(i,j)=zsec4(nxfield*(ny-j-1)+i+1)/ga
          if (xglobal) oro(nx-1,j)=oro(0,j)
20        continue
      endif

      call grib_release(igrib)
      goto 10                      !! READ NEXT LEVEL OR PARAMETER
*
* CLOSING OF INPUT DATA FILE
*
30    call grib_close_file(ifile)

*     error message if no fields found with correct first longitude in it
      if (gotGrid.eq.0) then
        print*,'***ERROR: input file needs to contain GRiB1 formatted'//
     &'messages'
        stop
      endif

      nuvz=iumax
      nwz =iwmax
      if(nuvz.eq.nlev_ec) nwz=nlev_ec+1

      if (nx.gt.nxmax) then
        write(*,*) 'FLEXTRA error: Too many grid points in x direction.'
        write(*,*) 'Reduce resolution of wind fields.'
        write(*,*) 'Or change parameter settings in file includepar.'
        write(*,*) nx,nxmax
        error=.true.
        return
      endif

      if (ny.gt.nymax) then
        write(*,*) 'FLEXTRA error: Too many grid points in y direction.'
        write(*,*) 'Reduce resolution of wind fields.'
        write(*,*) 'Or change parameter settings in file includepar.'
        write(*,*) ny,nymax
        error=.true.
        return
      endif

      if (nuvz.gt.nuvzmax) then
        write(*,*) 'FLEXTRA error: Too many u,v grid points in z '//
     +'direction.'
        write(*,*) 'Reduce resolution of wind fields.'
        write(*,*) 'Or change parameter settings in file includepar.'
        write(*,*) nuvz+1,nuvzmax
        error=.true.
        return
      endif

      if (nwz.gt.nwzmax) then
        write(*,*) 'FLEXTRA error: Too many w grid points in z '//
     +'direction.'
        write(*,*) 'Reduce resolution of wind fields.'
        write(*,*) 'Or change parameter settings in file includepar.'
        write(*,*) nwz,nwzmax
        error=.true.
        return
      endif

C Output of grid info
*********************

      write(*,*)
      write(*,*)
      write(*,'(a,2i7)') '# of vertical levels: ',nuvz,nwz
      write(*,*)
      write(*,'(a)') 'Mother domain:'
      write(*,'(a,f10.2,a1,f10.2,a,f10.2)') '  Longitude range: ',
     +xlon0,' to ',xlon0+(nx-1)*dx,'   Grid distance: ',dx
      write(*,'(a,f10.2,a1,f10.2,a,f10.2)') '  Latitude range:  ',
     +ylat0,' to ',ylat0+(ny-1)*dy,'   Grid distance: ',dy
      write(*,*)


C Compute often used aux variables to convert geografical into grid coord.
***************************************************************************

      xthelp=180./pi/r_earth/dx
      ythelp=180./pi/r_earth/dy


* CALCULATE VERTICAL DISCRETIZATION OF ECMWF MODEL
* PARAMETER akm,bkm DESCRIBE THE HYBRID "ETA" COORDINATE SYSTEM
* wheight(i) IS THE HEIGHT OF THE i-th MODEL HALF LEVEL (=INTERFACE BETWEEN
* 2 MODEL LEVELS) IN THE "ETA" SYSTEM

      numskip=nlev_ec-nuvz  ! number of ecmwf model layers not used
                            ! by trajectory model
      do 40 i=1,nwz
        j=numskip+i
        k=nlev_ec+1+numskip+i
        akm(nwz-i+1)=zsec2(j)
        bkm(nwz-i+1)=zsec2(k)
40      wheight(nwz-i+1)=akm(nwz-i+1)/p0+bkm(nwz-i+1)

* CALCULATION OF uvheight, akz, bkz
* akz,bkz ARE THE DISCRETIZATION PARAMETERS FOR THE MODEL LEVELS
* uvheight(i) IS THE HEIGHT OF THE i-th MODEL LEVEL IN THE "ETA" SYSTEM

      do 45 i=1,nuvz
        uvheight(i)=0.5*(wheight(i+1)+wheight(i))
        akz(i)=0.5*(akm(i+1)+akm(i))
        bkz(i)=0.5*(bkm(i+1)+bkm(i))
45      continue

C If vertical coordinates decrease with increasing altitude, multiply by -1.
C This means that also the vertical velocities have to be multiplied by -1.
****************************************************************************

      if (uvheight(1).lt.uvheight(nuvz)) then
        zdirect=1.
      else
        zdirect=-1.
        do 55 i=1,nuvz
55        uvheight(i)=zdirect*uvheight(i)
        do 65 i=1,nwz
65        wheight(i)=zdirect*wheight(i)
      endif


C Compute minimum and maximum height of modelling domain
********************************************************

      heightmin=max(uvheight(1),wheight(1))
      heightmax=min(uvheight(nuvz),wheight(nwz))


      return

999   write(*,*)
      write(*,*) ' ###########################################'//
     &           '###### '
      write(*,*) '       TRAJECTORY MODEL SUBROUTINE GRIDCHECK:'
      write(*,*) ' CAN NOT OPEN INPUT DATA FILE '//wfname(ifn)
      write(*,*) ' ###########################################'//
     &           '###### '
      write(*,*)
      write(*,'(a)') '!!! PLEASE INSERT A NEW CD-ROM AND   !!!'
      write(*,'(a)') '!!! PRESS ANY KEY TO CONTINUE...     !!!'
      write(*,'(a)') '!!! ...OR TERMINATE FLEXTRA PRESSING !!!'
      write(*,'(a)') '!!! THE "X" KEY...                   !!!'
      write(*,*)
      read(*,'(a)') opt
      if(opt.eq.'X') then
        error=.true.
      else
        goto 5
      endif

      return
      end
