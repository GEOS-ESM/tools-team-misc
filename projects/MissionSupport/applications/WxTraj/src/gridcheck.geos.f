      subroutine gridcheck(error)
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
*                                                                     *
***********************************************************************
*
      use netcdf

      include 'includepar'
      include 'includecom'
      include 'netcdf.inc'

      integer i,ifn,ifield,j,k,iumax,iwmax,numskip
      real sizesouth,sizenorth,xauxa
      logical error

!     parameters for netcdf
      integer ncid
      integer dimid, varid
      real*8  var(nwzmax)
      real*8  lon(nxmax),lat(nymax)

      real*4  xaux1,xaux2,yaux1,yaux2
!HSO  end


      error=.false.
      iumax=0
      iwmax=0

      if(ideltas.gt.0) then
        ifn=1
      else
        ifn=numbwf
      endif
!
! OPENING OF DATA FILE (NETCDF CODE)
!    
      call check(nf_open(path(3)(1:len(3))//trim(wfname(ifn)),
     +       NF_NOWRITE,ncid),
     +       "opening file" // trim(wfname(ifn)))


!
! Read Grid Dimensions
!

      call check(nf_inq_dimid(ncid, 'lon', dimid),
     +       "getting lon dimid")
      call check(nf_inq_dimlen(ncid, dimid,nxfield),
     +       "read nlon")

      call check(nf_inq_dimid(ncid, 'lat', dimid),
     +       "getting lat dimid")
      call check(nf_inq_dimlen(ncid, dimid,ny),
     +       "read nlat")

      call check(nf_inq_dimid(ncid, 'lev', dimid),
     +       "getting lev dimid")
      call check(nf_inq_dimlen(ncid, dimid,nlev_geos),
     +       "read nlev") 

! PHIS surface geopotential
      call check(nf_inq_varid(ncid,'PHIS',varid),
     +       "getting varid for PHIS")

      call check(nf_get_var(ncid,varid,oro),
     +       " reading PHIS")
! convert PHIS to oro [m]
      oro = oro/ga

! get lat/lon extent

      call check(nf_inq_varid(ncid,'lon',varid),
     +       "getting varid for lon")
      call check(nf_get_var(ncid,varid,lon),
     +       "reading lon")
! Westernmost Longitude
      xaux1 = real(lon(1))
! Easternmost Longitude
      xaux2 = real(lon(nxfield))

      call check(nf_inq_varid(ncid,'lat',varid),
     +       "getting varid for lat")
      call check(nf_get_var(ncid,varid,lat),
     +       "reading lat")
! Southernmost Latitude
      yaux1 = real(lat(1))
! Northernmost Latitude
      yaux2 = real(lat(ny))

      if (xaux1.gt.180.) xaux1=xaux1-360.0
      if (xaux2.gt.180.) xaux2=xaux2-360.0
      if (xaux1.lt.-180.) xaux1=xaux1+360.0
      if (xaux2.lt.-180.) xaux2=xaux2+360.0
      if (xaux2.lt.xaux1) xaux2=xaux2+360.0
      xlon0=xaux1
      ylat0=yaux1
      dx=(xaux2-xaux1)/float(nxfield-1)
      dy=(yaux2-yaux1)/float(ny-1)

! GEOS is always global
! and contains the poles, specify polar stereographic map
! projections using the stlmbr- and stcm2p-calls
!************************************************************

      nx=nxfield
      xglobal=.true.

      sglobal=.true.               ! field contains south pole
! Enhance the map scale by factor 3 (*2=6) compared to north-south
! map scale
      sizesouth=6.*(switchsouth+90.)/dy
      call stlmbr(southpolemap,-90.,0.)
      call stcm2p(southpolemap,0.,0.,switchsouth,0.,sizesouth,
     +    sizesouth,switchsouth,180.)
      switchsouthg=(switchsouth-ylat0)/dy

      nglobal=.true.               ! field contains north pole
! Enhance the map scale by factor 3 (*2=6) compared to north-south
! map scale
      sizenorth=6.*(90.-switchnorth)/dy
      call stlmbr(northpolemap,90.,0.)
      call stcm2p(northpolemap,0.,0.,switchnorth,0.,sizenorth,
     +    sizenorth,switchnorth,180.)
      switchnorthg=(switchnorth-ylat0)/dy

      iumax=max(iumax,nlev_geos)
      iwmax=max(iwmax,nlev_geos)

!
! CLOSING OF INPUT DATA FILE
!
      call check(nf_close(ncid),"closing" // trim(wfname(ifn)))

      nuvz=iumax
      nwz =iwmax
      if(nuvz.eq.nlev_geos) nwz=nlev_geos+1

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

! Output of grid info
!*********************

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


! Compute often used aux variables to convert geografical into grid coord.
!**************************************************************************

      xthelp=180./pi/r_earth/dx
      ythelp=180./pi/r_earth/dy


! CALCULATE VERTICAL DISCRETIZATION OF GEOS MODEL
! PARAMETER akm,bkm DESCRIBE THE HYBRID "ETA" COORDINATE SYSTEM
! wheight(i) IS THE HEIGHT OF THE i-th MODEL HALF LEVEL (=INTERFACE BETWEEN
! 2 MODEL LEVELS) IN THE "ETA" SYSTEM
      call check(nf_open('ak_bk.nc',
     +       NF_NOWRITE,ncid),
     +       "opening file ak_bk.nc")

      call check(nf_inq_varid(ncid,'ak',varid),
     +       "getting varid for ak")

      call check(nf_get_var(ncid,varid,var),
     +       " reading ak")
      akm = real(var)

      call check(nf_inq_varid(ncid,'bk',varid),
     +       "getting varid for bk")

      call check(nf_get_var(ncid,varid,var),
     +       " reading bk")     
      bkm = real(var)

      wheight=akm/p0+bkm
! CALCULATION OF uvheight, akz, bkz
! akz,bkz ARE THE DISCRETIZATION PARAMETERS FOR THE MODEL LEVELS
! uvheight(i) IS THE HEIGHT OF THE i-th MODEL LEVEL IN THE "ETA" SYSTEM

      do 45 i=1,nuvz
        uvheight(i)=0.5*(wheight(i+1)+wheight(i))
        akz(i)=0.5*(akm(i+1)+akm(i))
        bkz(i)=0.5*(bkm(i+1)+bkm(i))
45      continue

! If vertical coordinates decrease with increasing altitude, multiply by -1.
! This means that also the vertical velocities have to be multiplied by -1.
!****************************************************************************

      if (uvheight(1).lt.uvheight(nuvz)) then
        zdirect=1.
      else
        zdirect=-1.
        do i=1,nuvz
          uvheight(i)=zdirect*uvheight(i)
        end do
        do i=1,nwz
          wheight(i)=zdirect*wheight(i)
        end do
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

      return
      end
