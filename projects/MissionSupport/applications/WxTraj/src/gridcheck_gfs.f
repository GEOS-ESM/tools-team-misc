      subroutine gridcheck(oronew,error)
***********************************************************************
*                                                                     * 
*             TRAJECTORY MODEL SUBROUTINE GRIDCHECK                   *
*                                                                     *
***********************************************************************
*                                                                     * 
*             AUTHOR:      G. WOTAWA                                  *
*             DATE:        1997-08-06                                 *
*                                                                     * 
*             Update:      1998-12, global fields allowed, A. Stohl   *
*             Modification:2001-01, NCEP Pressure level data,         *
*                          G. Wotawa                                  * 
*             Sabine Eckhardt, Jan 2008 update to read GRIB2 with     *
*                              Grib Api                               *
*                                                                     * 
***********************************************************************
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
      include 'includepar'
      include 'includecom'
      include 'grib_api_f77.h'



      integer i,ifn,ifield,j,k,iumax,iwmax,numskip
      real sizesouth,sizenorth,xauxa
      real plev(nwzmax),help
      logical error,oronew

* VARIABLES AND ARRAYS NEEDED FOR GRIB DECODING

C dimension of isec2 at least (22+n), where n is the number of parallels or
C meridians in a quasi-regular (reduced) Gaussian or lat/long grid

C dimension of zsec2 at least (10+nn), where nn is the number of vertical
C coordinate parameters

      integer isec1(56),isec2(22+nxmax+nymax)
      real zsec4(jpunp)

!HSO  parameters for grib_api 
      integer ifile
      integer iret
      integer igrib
      integer*4 isize
      real*4  xaux1,xaux2,yaux1,yaux2
      real*8  xaux1in,xaux2in,yaux1in,yaux2in,zsecn4(jpunp*4)
      integer gribVer,parCat,parNum,typSurf,valSurf
!HSO  end

      integer i179,i180,i181
      real akm_usort(nwzmax)

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
      ifile=5
5     iret=grib_open_file(ifile,path(3)(1:len(3))//
     >trim(wfname(ifn)),'r')
      call grib_check(iret)
!     turn on support for multi fields messages */
      call grib_check(grib_multi_support_on())
!HSO 5    call pbopen(lunit,path(3)(1:len(3))//wfname(ifn),'r',ierr)
!     if(ierr.lt.0) goto 999
!HSO fin

      ifield=0   
10    ifield=ifield+1

*
* GET NEXT FIELDS
*
      iret = grib_new_from_file(ifile,igrib)
      if (igrib .eq. -1 )  then         
        if (iret .ne. -1) then
          call grib_check(iret)
          goto 999   ! ERROR DETECTED
        endif
        goto 30    ! EOF DETECTED
      endif



C Check whether we are on a little endian or on a big endian computer
*********************************************************************

c     if (inbuff(1).eq.1112101447) then         ! little endian, swap bytes
c       iswap=1+ilen/4
c       call swap32(inbuff,iswap)
c     else if (inbuff(1).ne.1196575042) then    ! big endian
c       stop 'subroutine gridcheck: corrupt GRIB data'
c     endif



!     first see if we read GRIB1 or GRIB2
      call grib_check(grib_get_int(  igrib,
     >'editionNumber',gribVer))

!     get the size and data of the values array
      call grib_check(grib_get_size(igrib,'values',isize))
      call grib_check(grib_get_real8_array(igrib,'values',zsecn4,isize))
      do  i=1,isize
        zsec4(i)=zsecn4(i)
      enddo

      if (gribVer.eq.1) then ! GRIB Edition 1

!     read the grib1 identifiers
      call grib_check(grib_get_int(  igrib,
     >'indicatorOfParameter',isec1(6)))
      call grib_check(grib_get_int(  igrib,
     >'indicatorOfTypeOfLevel',isec1(7)))
      call grib_check(grib_get_int(  igrib,
     >'level',isec1(8)))

      else ! GRIB Edition 2

!     read the grib2 identifiers
      call grib_check(grib_get_int(  igrib,
     >'parameterCategory',parCat))
      call grib_check(grib_get_int(  igrib,
     >'parameterNumber',parNum))
      call grib_check(grib_get_int(  igrib,
     >'typeOfFirstFixedSurface',typSurf))
      call grib_check(grib_get_int(  igrib,
     >'scaledValueOfFirstFixedSurface',valSurf))

!     convert to grib1 identifiers
      isec1(6)=-1
      isec1(7)=-1
      isec1(8)=-1
      if ((parCat.eq.2).and.(parNum.eq.2).and.(typSurf.eq.100)) then ! U
        isec1(6)=33          ! indicatorOfParameter
        isec1(7)=100         ! indicatorOfTypeOfLevel
        isec1(8)=valSurf/100 ! level, convert to hPa
      elseif ((parCat.eq.3).and.(parNum.eq.5).and.(typSurf.eq.1)) then ! TOPO 
        isec1(6)=7           ! indicatorOfParameter
        isec1(7)=1           ! indicatorOfTypeOfLevel
        isec1(8)=0
      elseif ((parCat.eq.0).and.(parNum.eq.0).and.(typSurf.eq.1)) then ! LSM
        isec1(6)=81          ! indicatorOfParameter
        isec1(7)=1           ! indicatorOfTypeOfLevel
        isec1(8)=0
      endif

      endif ! gribVer

      if(ifield.eq.1) then

!     get the required fields from section 2 
!     store compatible to gribex input
        call grib_check(grib_get_int(  igrib,
     >'numberOfPointsAlongAParallel',isec2(2)))
        call grib_check(grib_get_int(  igrib,
     >'numberOfPointsAlongAMeridian',isec2(3)))
        call grib_check(grib_get_real8(igrib,
     >'latitudeOfFirstGridPointInDegrees',yaux2in))
        call grib_check(grib_get_real8(igrib,
     >'longitudeOfFirstGridPointInDegrees',xaux1in))
        call grib_check(grib_get_real8(igrib,
     >'latitudeOfLastGridPointInDegrees',yaux1in))
        call grib_check(grib_get_real8(igrib,
     >'longitudeOfLastGridPointInDegrees',xaux2in))
      xaux1=xaux1in
      xaux2=xaux2in
      yaux1=yaux1in
      yaux2=yaux2in

c     if (ierr.ne.0) goto 10    ! ERROR DETECTED



        nxfield=isec2(2)
        ny=isec2(3)
        if((abs(xaux1).lt.eps).and.(xaux2.ge.359)) then ! NCEP DATA FROM 0 TO
          xaux1=-179.0                             ! 359 DEG EAST ->
          xaux2=-179.0+360.-360./float(nxfield)    ! TRANSFORMED TO -179
        endif                                      ! TO 180 DEG EAST
        if (xaux1.gt.180) xaux1=xaux1-360.0
        if (xaux2.gt.180) xaux2=xaux2-360.0
        if (xaux1.lt.-180) xaux1=xaux1+360.0
        if (xaux2.lt.-180) xaux2=xaux2+360.0
        if (xaux2.lt.xaux1) xaux2=xaux2+360.
        xlon0=xaux1
        ylat0=yaux1
        dx=(xaux2-xaux1)/float(nxfield-1)
        dy=(yaux2-yaux1)/float(ny-1)

        i179=nint(179./dx)
        i180=nint(179./dx)+1
        i181=i180+1

C Check whether fields are global
C If they contain the poles, specify polar stereographic map 
C projections using the stlmbr- and stcm2p-calls
************************************************************

        xauxa=abs(xaux2+dx-360.-xaux1)
        if (xauxa.lt.0.001) then
          nx=nxfield+1                 ! field is cyclic
          xglobal=.true.
        else
          nx=nxfield
          xglobal=.false.
        endif
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
      endif

     
      if((isec1(6).eq.007).and.(isec1(7).eq.001)) oronew=.true.
*      k=isec1(8)
*      if(isec1(6).eq. 33) iumax=max(iumax,nlev_ec-k+1)
*      if(isec1(6).eq. 39) iwmax=max(iwmax,nlev_ec-k+1) 
      if((isec1(6).eq.33).and.(isec1(7).eq.100)) then
        iumax=iumax+1
        plev(iumax)=float(isec1(8))*100.0
      endif

      if((isec1(6).eq.007).and.(isec1(7).eq.001)) then
* TOPOGRAPHY
        do 20 j=0,ny-1
          do 21 i=0,nxfield-1
            help=zsec4(nxfield*(ny-j-1)+i+1)
            if(i.le.180) then
              oro(i179+i,j)=help
            else
              oro(i-i181,j)=help
            endif
21        continue
20        continue
      endif

      if (igrib.ne.-1) then
        call grib_check(grib_release(igrib)) 
      endif

      goto 10                      !! READ NEXT LEVEL OR PARAMETER
      
30    continue

*
* CLOSING OF INPUT DATA FILE
*
      call grib_check(grib_close_file(ifile))

      nuvz=iumax
      nwz =iumax
      nlev_ec=iumax

      if (nx.gt.nxmax) then                         
      write(*,*) 'FLEXTRA error: Too many grid points in x direction.'
      write(*,*) 'Reduce resolution of wind fields (file GRIDSPEC).'
      error=.true.
      return
      endif

      if (ny.gt.nymax) then                         
      write(*,*) 'FLEXTRA error: Too many grid points in y direction.'
      write(*,*) 'Reduce resolution of wind fields (file GRIDSPEC).'
      error=.true.
      return
      endif

      if (nuvz.gt.nuvzmax) then                         
      write(*,*) 'FLEXTRA error: Too many u,v grid points in z '//
     +'direction.'
      write(*,*) 'Reduce resolution of wind fields (file GRIDSPEC).'
      error=.true.
      return
      endif

      if (nwz.gt.nwzmax) then                         
      write(*,*) 'FLEXTRA error: Too many w grid points in z '//
     +'direction.'
      write(*,*) 'Reduce resolution of wind fields (file GRIDSPEC).'
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
        akm_usort(nwz-i+1)=plev(nwz-i+1)
40      bkm(nwz-i+1)=0.0

*******************************
* change Sabine Eckhardt: akm should always be in descending order ... readwind adapted!
*******************************
          do 41 i=1,nwz
             if (akm_usort(1).gt.akm_usort(2)) then
                akm(i)=akm_usort(i)
             else
                akm(i)=akm_usort(nwz-i+1)
             endif
41        continue


* CALCULATION OF uvheight, akz, bkz
* akz,bkz ARE THE DISCRETIZATION PARAMETERS FOR THE MODEL LEVELS
* uvheight(i) IS THE HEIGHT OF THE i-th MODEL LEVEL IN THE "ETA" SYSTEM

      do 43 i=1,nwz
43      wheight(nwz-i+1)=akm(nwz-i+1)/p0+bkm(nwz-i+1)

      do 45 i=1,nuvz
        uvheight(i)=wheight(i)
        akz(i)=akm(i)
        bkz(i)=bkm(i)
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
      error=.true.

      return
      end
