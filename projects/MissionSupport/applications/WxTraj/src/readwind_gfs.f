      subroutine readwind(indj,n)
***********************************************************************
*                                                                     * 
*             TRAJECTORY MODEL SUBROUTINE READWIND                    *
*                                                                     *
***********************************************************************
*                                                                     * 
*             AUTHOR:      G. WOTAWA                                  *
*             DATE:        1997-08-05                                 *
*             LAST UPDATE: ----------                                 *
*             Update:      1998-07-29, global fields allowed          *
*             A. Stohl, G. Wotawa                                     *
*             Update:      2001-01-05 NCEP Data Pressure levels       * 
*                                                                     * 
***********************************************************************
*                                                                     *
* DESCRIPTION:                                                        *
*                                                                     *
* READING OF ECMWF METEOROLOGICAL FIELDS FROM INPUT DATA FILES. THE   *
* INPUT DATA FILES ARE EXPECTED TO BE AVAILABLE IN GRIB CODE          *
*                                                                     *
* INPUT:                                                              *
* indj               indicates number of the wind field to be read in *
* n                  temporal index for meteorological fields (1 to 3)*
*                                                                     *
* IMPORTANT VARIABLES FROM COMMON BLOCK:                              *
*                                                                     *
* wfname             File name of data to be read in                  *
* nxfield,ny,nuvz,nwz     expected field dimensions                   *
* nlev_ec            number of vertical levels ecmwf model            *
* uu,vv,ww           wind fields                                      *
* tt,qq              temperature and specific humidity                *
* ps                 surface pressure                                 *
*                                                                     *
***********************************************************************

      include 'includepar'
      include 'includecom'
      include 'grib_api_f77.h'

      integer ii,indj,i,j,k,n,ifield,iumax
      integer ix,jy,induvz,indwz,numpt,numpu,numpv,numpw,numprh
      real help,temp,plev,ew,elev

* VARIABLES AND ARRAYS NEEDED FOR GRIB DECODING

C dimension of isec2 at least (22+n), where n is the number of parallels or
C meridians in a quasi-regular (reduced) Gaussian or lat/long grid

C dimension of zsec2 at least (10+nn), where nn is the number of vertical
C coordinate parameters

      integer isec1(56),isec2(22+nxmax+nymax)
      real zsec4(jpunp)
      real xaux,yaux,xaux0,yaux0
      real ylat,xlon,wdummy,ffpol,ddpol,xlonr
      real uuaux,vvaux,uupolaux,vvpolaux

      logical error

!     parameters for grib_api 
      integer ifile
      integer iret
      integer igrib
      integer*4 isize
      integer gribVer,parCat,parNum,typSurf,valSurf
      real*8 zsecn4(jpunp*4) 
      real*8 xauxin,yauxin

      integer i179,i180,i181

      iumax=0
*
* OPENING OF DATA FILE (GRIB CODE)
*
*
* OPENING OF DATA FILE (GRIB CODE)
*
!HSO
!     print*,'reading winds from ',path(3)(1:len(3))
!    >//trim(wfname(indj)),'|'
5     iret=grib_open_file(ifile,path(3)(1:len(3))
     >//trim(wfname(indj)),'r')
      call grib_check(iret)
!     turn on support for multi fields messages
      call grib_check(grib_multi_support_on())


      numpt=0
      numpu=0
      numpv=0
      numpw=0
      numprh=0
      ifield=0
10    ifield=ifield+1
*
* GET NEXT FIELDS
*

      iret=grib_new_from_file(ifile,igrib)      
      if (igrib .eq. -1 )  then         
        if (iret .ne. -1) then
          call grib_check(iret)
          goto 888   ! ERROR DETECTED
        endif
        goto 50    ! EOF DETECTED
      endif

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
      if ((parCat.eq.0).and.(parNum.eq.0).and.(typSurf.eq.100)) then ! T
        isec1(6)=11          ! indicatorOfParameter
        isec1(7)=100         ! indicatorOfTypeOfLevel
        isec1(8)=valSurf/100 ! level, convert to hPa
      elseif ((parCat.eq.2).and.(parNum.eq.2).and.(typSurf.eq.100)) then ! U
        isec1(6)=33          ! indicatorOfParameter
        isec1(7)=100         ! indicatorOfTypeOfLevel
        isec1(8)=valSurf/100 ! level, convert to hPa
      elseif ((parCat.eq.2).and.(parNum.eq.3).and.(typSurf.eq.100)) then ! V
        isec1(6)=34          ! indicatorOfParameter
        isec1(7)=100         ! indicatorOfTypeOfLevel
        isec1(8)=valSurf/100 ! level, convert to hPa
      elseif ((parCat.eq.2).and.(parNum.eq.8).and.(typSurf.eq.100)) then ! W
        isec1(6)=39          ! indicatorOfParameter
        isec1(7)=100         ! indicatorOfTypeOfLevel
        isec1(8)=valSurf/100 ! level, convert to hPa
      elseif ((parCat.eq.1).and.(parNum.eq.1).and.(typSurf.eq.100)) then ! RH
        isec1(6)=52          ! indicatorOfParameter
        isec1(7)=100         ! indicatorOfTypeOfLevel
        isec1(8)=valSurf/100 ! level, convert to hPa
      elseif ((parCat.eq.1).and.(parNum.eq.1).and.(typSurf.eq.103)) then ! RH2
        isec1(6)=52          ! indicatorOfParameter
        isec1(7)=105         ! indicatorOfTypeOfLevel
        isec1(8)=2
      elseif ((parCat.eq.0).and.(parNum.eq.0).and.(typSurf.eq.103)) then ! T2
        isec1(6)=11          ! indicatorOfParameter
        isec1(7)=105         ! indicatorOfTypeOfLevel
        isec1(8)=2
      elseif ((parCat.eq.2).and.(parNum.eq.2).and.(typSurf.eq.103)) then ! U10
        isec1(6)=33          ! indicatorOfParameter
        isec1(7)=105         ! indicatorOfTypeOfLevel
        isec1(8)=10
      elseif ((parCat.eq.2).and.(parNum.eq.3).and.(typSurf.eq.103)) then ! V10
        isec1(6)=34          ! indicatorOfParameter
        isec1(7)=105         ! indicatorOfTypeOfLevel
        isec1(8)=10
      elseif ((parCat.eq.3).and.(parNum.eq.1).and.(typSurf.eq.101)) then ! SLP
        isec1(6)=2           ! indicatorOfParameter
        isec1(7)=102         ! indicatorOfTypeOfLevel
        isec1(8)=0
      elseif ((parCat.eq.3).and.(parNum.eq.0).and.(typSurf.eq.1)) then ! SP
        isec1(6)=1           ! indicatorOfParameter
        isec1(7)=1           ! indicatorOfTypeOfLevel
        isec1(8)=0
      elseif ((parCat.eq.1).and.(parNum.eq.13).and.(typSurf.eq.1)) then ! SNOW
        isec1(6)=66          ! indicatorOfParameter
        isec1(7)=1           ! indicatorOfTypeOfLevel
        isec1(8)=0
      elseif ((parCat.eq.0).and.(parNum.eq.0).and.(typSurf.eq.104)) then ! T sigma 0
        isec1(6)=11          ! indicatorOfParameter
        isec1(7)=107         ! indicatorOfTypeOfLevel
        isec1(8)=0.995       ! lowest sigma level
      elseif ((parCat.eq.2).and.(parNum.eq.2).and.(typSurf.eq.104)) then ! U sigma 0
        isec1(6)=33          ! indicatorOfParameter
        isec1(7)=107         ! indicatorOfTypeOfLevel
        isec1(8)=0.995       ! lowest sigma level
      elseif ((parCat.eq.2).and.(parNum.eq.3).and.(typSurf.eq.104)) then ! V sigma 0
        isec1(6)=34          ! indicatorOfParameter
        isec1(7)=107         ! indicatorOfTypeOfLevel
        isec1(8)=0.995       ! lowest sigma level
      elseif ((parCat.eq.3).and.(parNum.eq.5).and.(typSurf.eq.1)) then ! TOPO 
        isec1(6)=7           ! indicatorOfParameter
        isec1(7)=1           ! indicatorOfTypeOfLevel
        isec1(8)=0
      elseif ((parCat.eq.0).and.(parNum.eq.0).and.(typSurf.eq.1)) then ! LSM
        isec1(6)=81          ! indicatorOfParameter
        isec1(7)=1           ! indicatorOfTypeOfLevel
        isec1(8)=0
      elseif ((parCat.eq.3).and.(parNum.eq.196).and.(typSurf.eq.1)) then ! BLH
        isec1(6)=221         ! indicatorOfParameter
        isec1(7)=1           ! indicatorOfTypeOfLevel
        isec1(8)=0
      elseif ((parCat.eq.1).and.(parNum.eq.7).and.(typSurf.eq.1)) then ! LSP/TP
        isec1(6)=62          ! indicatorOfParameter
        isec1(7)=1           ! indicatorOfTypeOfLevel
        isec1(8)=0
      elseif ((parCat.eq.1).and.(parNum.eq.196).and.(typSurf.eq.1)) then ! CP
        isec1(6)=63          ! indicatorOfParameter
        isec1(7)=1           ! indicatorOfTypeOfLevel
        isec1(8)=0
      endif

      endif ! gribVer


C Check whether we are on a little endian or on a big endian computer
*********************************************************************

c     if (inbuff(1).eq.1112101447) then         ! little endian, swap bytes
c       iswap=1+ilen/4
c       call swap32(inbuff,iswap)
c     else if (inbuff(1).ne.1196575042) then    ! big endian
c       stop 'subroutine gridcheck: corrupt GRIB data'
c     endif

c     if (ierr.ne.0) goto 10    ! ERROR DETECTED

      if(ifield.eq.1) then

!     get the required fields from section 2 
!     store compatible to gribex input
      call grib_check(grib_get_int(  igrib,
     >'numberOfPointsAlongAParallel',isec2(2)))
      call grib_check(grib_get_int(  igrib,
     >'numberOfPointsAlongAMeridian',isec2(3)))
      call grib_check(grib_get_real8(igrib,
     >'longitudeOfFirstGridPointInDegrees',xauxin))
      call grib_check(grib_get_real8(igrib,
     >'latitudeOfLastGridPointInDegrees',yauxin))
      xaux=xauxin
      yaux=yauxin

* CHECK GRID SPECIFICATIONS

         if(isec2(2).ne.nxfield) stop 'READWIND: NX NOT CONSISTENT'
        if(isec2(3).ne.ny) stop 'READWIND: NY NOT CONSISTENT'
        if(xaux.eq.0.) xaux=-179.0     ! NCEP DATA
        xaux0=xlon0
        yaux0=ylat0
        if(xaux.lt.0.) xaux=xaux+360.
        if(yaux.lt.0.) yaux=yaux+360.
        if(xaux0.lt.0.) xaux0=xaux0+360.
        if(yaux0.lt.0.) yaux0=yaux0+360.
        if(abs(xaux-xaux0).gt.eps) 
     &    stop 'READWIND: LOWER LEFT LONGITUDE NOT CONSISTENT'
        if(abs(yaux-yaux0).gt.eps)
     &    stop 'READWIND: LOWER LEFT LATITUDE NOT CONSISTENT'
      endif
!HSO end of edits

      i179=nint(179./dx)
      i180=nint(179./dx)+1
      i181=i180+1

      do 20 j=0,ny-1
        do 20 i=0,nxfield-1

          if((isec1(6).eq.011).and.(isec1(7).eq.100)) then
* TEMPERATURE
           if((i.eq.0).and.(j.eq.0)) then
                do 21 ii=1,nuvz
                  if ((isec1(8)*100.0).eq.akz(ii)) numpt=ii
21              continue
            endif
            help=zsec4(nxfield*(ny-j-1)+i+1)
            if(i.le.i180) then
              tt(i179+i,j,numpt,n)=help
            else
              tt(i-i181,j,numpt,n)=help
            endif
          endif

          if((isec1(6).eq.033).and.(isec1(7).eq.100)) then
* U VELOCITY
             if((i.eq.0).and.(j.eq.0)) then
                do 22 ii=1,nuvz
                  if ((isec1(8)*100.0).eq.akz(ii)) numpu=ii
22              continue
            endif

            help=zsec4(nxfield*(ny-j-1)+i+1)
            if(i.le.i180) then
              uu(i179+i,j,numpu,n)=help
            else
              uu(i-i181,j,numpu,n)=help
            endif
          endif

          if((isec1(6).eq.034).and.(isec1(7).eq.100)) then
* V VELOCITY
             if((i.eq.0).and.(j.eq.0)) then
                do 23 ii=1,nuvz
                  if ((isec1(8)*100.0).eq.akz(ii)) numpv=ii
23              continue
            endif
            help=zsec4(nxfield*(ny-j-1)+i+1)
            if(i.le.i180) then
              vv(i179+i,j,numpv,n)=help
            else
              vv(i-i181,j,numpv,n)=help
            endif
          endif

          if((isec1(6).eq.039).and.(isec1(7).eq.100)) then
* W VELOCITY
             if((i.eq.0).and.(j.eq.0)) then
                do 25 ii=1,nuvz
                  if ((isec1(8)*100.0).eq.akz(ii)) numpw=ii
25              continue
            endif
            help=zsec4(nxfield*(ny-j-1)+i+1)
            if(i.le.i180) then
              ww(i179+i,j,numpw,n)=help
            else
              ww(i-i181,j,numpw,n)=help
            endif
          endif

          if((isec1(6).eq.052).and.(isec1(7).eq.100)) then
* RELATIVE HUMIDITY -> CONVERT TO SPECIFIC HUMIDITY
          if((i.eq.0).and.(j.eq.0)) then
                do 24 ii=1,nuvz
                  if ((isec1(8)*100.0).eq.akz(ii)) numprh=ii
24              continue
            endif
            help=zsec4(nxfield*(ny-j-1)+i+1)
            if(i.le.i180) then
              qq(i179+i,j,numprh,n)=help
            else
              qq(i-i181,j,numprh,n)=help
            endif
          endif

          if((isec1(6).eq.001).and.(isec1(7).eq.001)) then
* SURFACE PRESSURE
            help=zsec4(nxfield*(ny-j-1)+i+1)
            if(i.le.i180) then
              ps(i179+i,j,1,n)=help
            else
              ps(i-i181,j,1,n)=help
            endif
          endif

          if((isec1(6).eq.007).and.(isec1(7).eq.001)) then
* TOPOGRAPHY
            help=zsec4(nxfield*(ny-j-1)+i+1)
            if(i.le.i180) then
              oro(i179+i,j)=help
            else
              oro(i-i181,j)=help
            endif
          endif

20        continue

          if((isec1(6).eq.33).and.(isec1(7).eq.100)) iumax=iumax+1

      if (igrib.ne.-1) then
        call grib_check(grib_release(igrib)) 
      endif

      goto 10                      !! READ NEXT LEVEL OR PARAMETER

50    continue
*
* CLOSING OF INPUT DATA FILE
*
      call grib_check(grib_close_file(ifile))



* TRANSFORM RH TO SPECIFIC HUMIDITY AS NEEDED

      do 65 j=0,ny-1
        do 65 i=0,nxfield-1
          do 65 k=1,nuvz
            help=qq(i,j,k,n)
            temp=tt(i,j,k,n)
            plev=akm(k)
            elev=ew(temp)*help/100.0
            qq(i,j,k,n)=xmwml*(elev/(plev-((1.0-xmwml)*elev)))
65    continue

C For global fields, assign rightmost grid point the value of the
C leftmost point
*****************************************************************

      if (xglobal) then
        do 70 j=0,ny-1
          oro(nx-1,j)=oro(0,j)
          ps(nx-1,j,1,n)=ps(0,j,1,n)
          do 71 induvz=1,nuvz
            tt(nx-1,j,induvz,n)=tt(0,j,induvz,n)
            qq(nx-1,j,induvz,n)=qq(0,j,induvz,n)
            uu(nx-1,j,induvz,n)=uu(0,j,induvz,n)
71          vv(nx-1,j,induvz,n)=vv(0,j,induvz,n)
          do 70 indwz=1,nwz
70          ww(nx-1,j,indwz,n)=ww(0,j,indwz,n)
      endif


C If north pole is in the domain, calculate wind velocities in polar
C stereographic coordinates
********************************************************************

      if (nglobal) then
        do 74 jy=int(switchnorthg)-2,ny-1
          ylat=ylat0+float(jy)*dy
          do 74 ix=0,nx-1
            xlon=xlon0+float(ix)*dx
            do 74 induvz=1,nuvz
74            call cc2gll(northpolemap,ylat,xlon,uu(ix,jy,induvz,n),
     +        vv(ix,jy,induvz,n),uupol(ix,jy,induvz,n),
     +        vvpol(ix,jy,induvz,n))


        do 76 induvz=1,nuvz

* CALCULATE FFPOL, DDPOL FOR CENTRAL GRID POINT
          xlon=xlon0+float(nx/2-1)*dx
          xlonr=xlon*pi/180.
          ffpol=sqrt(uu(nx/2-1,ny-1,induvz,n)**2+
     &               vv(nx/2-1,ny-1,induvz,n)**2)
          if(vv(nx/2-1,ny-1,induvz,n).lt.0.) then
            if(vv(nx/2-1,ny-1,induvz,n).gt.-0.00001)
     &         vv(nx/2-1,ny-1,induvz,n)=-0.00001
            ddpol=atan(uu(nx/2-1,ny-1,induvz,n)/
     &                 vv(nx/2-1,ny-1,induvz,n))-xlonr
          else
            if(vv(nx/2-1,ny-1,induvz,n).lt. 0.00001)
     &         vv(nx/2-1,ny-1,induvz,n)= 0.00001
            ddpol=pi+atan(uu(nx/2-1,ny-1,induvz,n)/
     &                    vv(nx/2-1,ny-1,induvz,n))-xlonr
          endif
          if(ddpol.lt.0.) ddpol=2.0*pi+ddpol
          if(ddpol.gt.2.0*pi) ddpol=ddpol-2.0*pi

* CALCULATE U,V FOR 180 DEG, TRANSFORM TO POLAR STEREOGRAPHIC GRID
          xlon=180.0
          xlonr=xlon*pi/180.
          ylat=90.0
          uuaux=-ffpol*sin(xlonr+ddpol)
          vvaux=-ffpol*cos(xlonr+ddpol)
          call cc2gll(northpolemap,ylat,xlon,uuaux,vvaux,uupolaux,
     +      vvpolaux)

          jy=ny-1
          do 76 ix=0,nx-1
            uupol(ix,jy,induvz,n)=uupolaux
            vvpol(ix,jy,induvz,n)=vvpolaux
76      continue


* Fix: Set W at pole to the zonally averaged W of the next equator-
* ward parallel of latitude
 
      do 85 indwz=1,nwz
          wdummy=0.
          jy=ny-2
          do 80 ix=0,nx-1
80          wdummy=wdummy+ww(ix,jy,indwz,n)
          wdummy=wdummy/float(nx)
          jy=ny-1
          do 85 ix=0,nx-1
85          ww(ix,jy,indwz,n)=wdummy

      endif

C If south pole is in the domain, calculate wind velocities in polar
C stereographic coordinates
********************************************************************

      if (sglobal) then
        do 77 jy=0,int(switchsouthg)+3
          ylat=ylat0+float(jy)*dy
          do 77 ix=0,nx-1
            xlon=xlon0+float(ix)*dx
            do 77 induvz=1,nuvz
77            call cc2gll(southpolemap,ylat,xlon,uu(ix,jy,induvz,n),
     +        vv(ix,jy,induvz,n),uupol(ix,jy,induvz,n),
     +        vvpol(ix,jy,induvz,n))

        do 79 induvz=1,nuvz

* CALCULATE FFPOL, DDPOL FOR CENTRAL GRID POINT
          xlon=xlon0+float(nx/2-1)*dx
          xlonr=xlon*pi/180.
          ffpol=sqrt(uu(nx/2-1,0,induvz,n)**2+
     &               vv(nx/2-1,0,induvz,n)**2)
          if(vv(nx/2-1,0,induvz,n).lt.0.) then
            if(vv(nx/2-1,0,induvz,n).gt.-0.00001) 
     &        vv(nx/2-1,0,induvz,n)=-0.00001
            ddpol=atan(uu(nx/2-1,0,induvz,n)/
     &                 vv(nx/2-1,0,induvz,n))+xlonr
          else
            if(vv(nx/2-1,0,induvz,n).lt. 0.00001) 
     &        vv(nx/2-1,0,induvz,n)= 0.00001
            ddpol=pi+atan(uu(nx/2-1,0,induvz,n)/
     &                    vv(nx/2-1,0,induvz,n))+xlonr
          endif
          if(ddpol.lt.0.) ddpol=2.0*pi+ddpol
          if(ddpol.gt.2.0*pi) ddpol=ddpol-2.0*pi

* CALCULATE U,V FOR 180 DEG, TRANSFORM TO POLAR STEREOGRAPHIC GRID
          xlon=180.0
          xlonr=xlon*pi/180.
          ylat=-90.0
          uuaux=+ffpol*sin(xlonr-ddpol)
          vvaux=-ffpol*cos(xlonr-ddpol)
          call cc2gll(northpolemap,ylat,xlon,uuaux,vvaux,uupolaux,
     +      vvpolaux)

          jy=0
          do 79 ix=0,nx-1
            uupol(ix,jy,induvz,n)=uupolaux
79          vvpol(ix,jy,induvz,n)=vvpolaux


* Fix: Set W at pole to the zonally averaged W of the next equator-
* ward parallel of latitude
 
        do 95 indwz=1,nwz
          wdummy=0.
          jy=1
          do 90 ix=0,nx-1
90          wdummy=wdummy+ww(ix,jy,indwz,n)
          wdummy=wdummy/float(nx)
          jy=0
          do 95 ix=0,nx-1
95          ww(ix,jy,indwz,n)=wdummy
      endif

      WRITE(*,*) WFTIME(INDJ),' SEC ',WFTIME(INDJ)/3600,' HRS ',
     +WFTIME(INDJ)/3600/24,' DAYS  ', WFNAME(INDJ)
      if(iumax.ne.nuvz) stop 'READWIND: NUVZ NOT CONSISTENT'
      if(iumax.ne.nwz)  stop 'READWIND: NWZ NOT CONSISTENT'


C Calculate potential temperature and potential vorticity on whole grid
***********************************************************************

      call calcpv(n)


      return    
888   write(*,*) ' #### TRAJECTORY MODEL ERROR! WINDFIELD       #### ' 
      write(*,*) ' #### ',wfname(indj),'                    #### '
      write(*,*) ' #### IS NOT GRIB FORMAT !!!                  #### '
      stop 'Execution terminated'

999   write(*,*) ' #### TRAJECTORY MODEL ERROR! WINDFIELD       #### ' 
      write(*,*) ' #### ',wfname(indj),'                    #### '
      write(*,*) ' #### CANNOT BE OPENED !!!                    #### '
      error=1

      end
