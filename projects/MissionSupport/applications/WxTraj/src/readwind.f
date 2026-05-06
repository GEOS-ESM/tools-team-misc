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
*                          2011-06, implemented reading of grib2 format*
*                                   analog to FLEXPART 8.22 routines  *
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

      use GRIB_API

      include 'includepar'
      include 'includecom'

!HSO  parameters for grib_api
      integer ifile
      integer iret
      integer igrib
      integer gribVer,parCat,parNum,typSurf,valSurf,discipl
      integer gotGrid
      character*24 gribErrorMsg
      character*20 gribFunction
!HSO  end

      integer indj,i,j,k,n,levdiff2,ifield,iumax,iwmax,lunit
      integer ix,jy,induvz,indwz

* VARIABLES AND ARRAYS NEEDED FOR GRIB DECODING

C dimension of isec2 at least (22+n), where n is the number of parallels or
C meridians in a quasi-regular (reduced) Gaussian or lat/long grid

C dimension of zsec2 at least (10+nn), where nn is the number of vertical
C coordinate parameters

      integer isec0(2),isec1(56),isec2(22+nxmax+nymax),isec3(2)
      integer isec4(64),inbuff(jpack),ilen,iswap,ierr,iword
      real*4 nsss(0:nxmax-1,0:nymax-1),ewss(0:nxmax-1,0:nymax-1)
      real zsec2(60+2*nuvzmax),zsec3(2),zsec4(jpunp)
      real xaux,yaux,xaux0,yaux0
      real*8 xauxin,yauxin
      real ylat,xlon,wdummy,ffpol,ddpol,xlonr
      real uuaux,vvaux,uupolaux,vvpolaux

      character*1 yoper,opt
      logical error

!HSO  grib api error messages
      data gribErrorMsg/'Error reading grib file'/
      data gribFunction/'readwind'/

      data yoper/'D'/

      levdiff2=nlev_ec-nwz+1
      iumax=0
      iwmax=0
*
* OPENING OF DATA FILE (GRIB CODE)
*

5     call grib_open_file(ifile,path(3)(1:len(3))
     >//trim(wfname(indj)),'r',iret)
      if (iret.ne.GRIB_SUCCESS) then
        goto 888   ! ERROR DETECTED
      endif

      gotGrid=0
      ifield=0
10    ifield=ifield+1
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
        call grib_get_int(igrib,'typeOfFirstFixedSurface',typSurf,iret)
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
          isec1(6)=165         ! indicatorOfParamete
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
        elseif((parCat.eq.2).and.(parNum.eq.17)) then ! EWSS
          isec1(6)=180         ! indicatorOfParameter
        elseif((parCat.eq.2).and.(parNum.eq.18)) then ! NSSS
          isec1(6)=181         ! indicatorOfParameter
        elseif((parCat.eq.3).and.(parNum.eq.4)) then ! ORO
          isec1(6)=129         ! indicatorOfParameter
        elseif((parCat.eq.3).and.(parNum.eq.7)) then ! SDO
          isec1(6)=160         ! indicatorOfParameter
        elseif((discipl.eq.2).and.(parCat.eq.0).and.(parNum.eq.0).and.
     +   (typSurf.eq.1)) then ! LSM
          isec1(6)=172         ! indicatorOfParameter
        else
          print*,'***ERROR: undefined GRiB2 message found!',discipl,
     +     parCat,parNum,typSurf
        endif
      endif

!HSO  get the size and data of the values array
      if (isec1(6).ne.-1) then
        call grib_get_real4_array(igrib,'values',zsec4,iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
      endif

!HSO  get the required fields from section 2 in a gribex compatible manner
      if (ifield.eq.1) then
        call grib_get_int(igrib,'numberOfPointsAlongAParallel',
     +       isec2(2),iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
        call grib_get_int(igrib,'numberOfPointsAlongAMeridian',
     +       isec2(3),iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
        call grib_get_int(igrib,'numberOfVerticalCoordinateValues',
     +     isec2(12))
        call grib_check(iret,gribFunction,gribErrorMsg)
* CHECK GRID SPECIFICATIONS
        if(isec2(2).ne.nxfield) stop 'READWIND: NX NOT CONSISTENT'
        if(isec2(3).ne.ny) stop 'READWIND: NY NOT CONSISTENT'
        if(isec2(12)/2-1.ne.nlev_ec)
     +    stop 'READWIND: VERTICAL DISCRETIZATION NOT CONSISTENT'
      endif ! ifield

!HSO  get the second part of the grid dimensions only from GRiB1 messages
      if ((gribVer.eq.1).and.(gotGrid.eq.0)) then
        call grib_get_real8(igrib,'longitudeOfFirstGridPointInDegrees',
     >     xauxin,iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
        call grib_get_real8(igrib,'latitudeOfLastGridPointInDegrees',
     >     yauxin,iret)
        call grib_check(iret,gribFunction,gribErrorMsg)
        xaux=xauxin
        yaux=yauxin
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
        gotGrid=1
      endif ! gotGrid

      do 20 j=0,ny-1
        do 20 i=0,nxfield-1
          k=isec1(8)
          if(isec1(6).eq.130) tt(i,j,nlev_ec-k+1,n)=    !! TEMPERATURE
     &                        zsec4(nxfield*(ny-j-1)+i+1)
          if(isec1(6).eq.131) uu(i,j,nlev_ec-k+1,n)=    !! U VELOCITY
     &                        zsec4(nxfield*(ny-j-1)+i+1)
          if(isec1(6).eq.132) vv(i,j,nlev_ec-k+1,n)=    !! V VELOCITY
     &                        zsec4(nxfield*(ny-j-1)+i+1)
          if(isec1(6).eq.133) then
            qq(i,j,nlev_ec-k+1,n)=                      !! SPEC. HUMIDITY
     &                        zsec4(nxfield*(ny-j-1)+i+1)
            if (qq(i,j,nlev_ec-k+1,n).lt.0.) qq(i,j,nlev_ec-k+1,n)=0.
c             this is necessary because the gridded data may contain
c             spurious negative values
          endif
          if(isec1(6).eq.134) ps(i,j,1,n)=              !! SURF. PRESS.
     &                        zsec4(nxfield*(ny-j-1)+i+1)
          if(isec1(6).eq.135) ww(i,j,nlev_ec-k+1,n)=    !! W VELOCITY
     &                        zsec4(nxfield*(ny-j-1)+i+1)
          if(isec1(6).eq.129) oro(i,j)=                 !! ECMWF OROGRAPHY
     &                        zsec4(nxfield*(ny-j-1)+i+1)/ga
          if(isec1(6).eq.131) iumax=max(iumax,nlev_ec-k+1)
          if(isec1(6).eq.135) iwmax=max(iwmax,nlev_ec-k+1)

20        continue

      call grib_release(igrib)
      goto 10                      !! READ NEXT LEVEL OR PARAMETER
*
* CLOSING OF INPUT DATA FILE
*
50    call grib_close_file(ifile)

*     error message if no fields found with correct first longitude in it
      if (gotGrid.eq.0) then
        print*,'***ERROR: input file needs to contain GRiB1 formatted'//
     &'messages'
        stop
      endif

      if(levdiff2.eq.0) then
        iwmax=nlev_ec+1
        do 60 j=0,ny-1
          do 60 i=0,nx-1
60          ww(i,j,nlev_ec+1,n)=0.
      endif


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
            ddpol=atan(uu(nx/2-1,ny-1,induvz,n)/
     &                 vv(nx/2-1,ny-1,induvz,n))-xlonr
          else
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
            ddpol=atan(uu(nx/2-1,0,induvz,n)/
     &                 vv(nx/2-1,0,induvz,n))+xlonr
          else
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

      if(iumax.ne.nuvz) stop 'READWIND: NUVZ NOT CONSISTENT'
      if(iwmax.ne.nwz)  stop 'READWIND: NWZ NOT CONSISTENT'


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
      write(*,*)
      write(*,'(a)') '!!! PLEASE INSERT A NEW CD-ROM AND   !!!'
      write(*,'(a)') '!!! PRESS ANY KEY TO CONTINUE...     !!!'
      write(*,'(a)') '!!! ...OR TERMINATE FLEXTRA PRESSING !!!'
      write(*,'(a)') '!!! THE "X" KEY...                   !!!'
      write(*,'(a)') '!!! PLEASE CHECK CD-ROM LABEL AND    !!!'
      write(*,'(a)') '!!! CORRECT FILE "PATHNAMES"...      !!!'
      write(*,*)
      read(*,'(a)') opt
      if(opt.eq.'X') then
        stop 'Execution terminated'
      else
        call readpaths(error)
        if(error)
     &    stop 'Error reading "pathnames" --> execution terminated'
        goto 5
      endif

      end
