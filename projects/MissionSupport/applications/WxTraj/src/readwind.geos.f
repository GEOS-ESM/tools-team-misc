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
! DESCRIPTION:                                                        *
!                                                                     *
! READING OF ECMWF METEOROLOGICAL FIELDS FROM INPUT DATA FILES. THE   *
! INPUT DATA FILES ARE EXPECTED TO BE AVAILABLE IN GRIB CODE          *
!                                                                     *
! INPUT:                                                              *
! indj               indicates number of the wind field to be read in *
! n                  temporal index for meteorological fields (1 to 3)*
!                                                                     *
! IMPORTANT VARIABLES FROM COMMON BLOCK:                              *
!                                                                     *
! wfname             File name of data to be read in                  *
! nxfield,ny,nuvz,nwz     expected field dimensions                   *
! nlev_geos            number of vertical levels geos model           *
! uu,vv,ww           wind fields                                      *
! tt,qq              temperature and specific humidity                *
! ps                 surface pressure                                 *
! oro                orography (PHIS/gravity)                         *
! delp               layer pressure thickness                         *
!                                                                     *
!**********************************************************************

      use netcdf

      include 'includepar'
      include 'includecom'
      include 'netcdf.inc'

!     parameters for netcdf
      integer ncid
      integer varid
      character*24 varname

      integer indj,i,j,k,z,n,levdiff2,ifield,iumax,iwmax,lunit
      integer ix,jy,induvz,indwz
      integer gotGrid

      real xaux,yaux,xaux0,yaux0
      character*24 attr
      real ylat,xlon,wdummy,ffpol,ddpol,xlonr
      real uuaux,vvaux,uupolaux,vvpolaux
      real var(0:nxmax-1,0:nymax-1,nuvzmax,1)
      real*8 lon(nxmax),lat(nymax)

      logical error

      levdiff2=nlev_geos-nwz+1

!
! OPENING OF DATA FILE (NETCDF CODE)
!

      call check(nf_open(path(3)(1:len(3))//trim(wfname(indj)),
     +       NF_NOWRITE,ncid),
     +       "opening file" // trim(wfname(indj)))

!
! Read Inputs
!

! U
      call check(nf_inq_varid(ncid,'U',varid),
     +       "getting varid for U")

      call check(nf_get_var(ncid,varid,var),
     +       " reading U")
! flip Z direction
      do 10 k=1,nlev_geos
          z = nlev_geos - k + 1
          uu(:,:,k,n) = var(:,:,z,1)
10    continue
! V
      call check(nf_inq_varid(ncid,'V',varid),
     +       "getting varid for V")

      call check(nf_get_var(ncid,varid,var),
     +       " reading V")
! flip Z direction
      do 11 k=1,nlev_geos
          z = nlev_geos - k + 1
          vv(:,:,k,n) = var(:,:,z,1)
11    continue     
! OMEGA [W]
      call check(nf_inq_varid(ncid,'OMEGA',varid),
     +       "getting varid for OMEGA")

      call check(nf_get_var(ncid,varid,var),
     +       " reading OMEGA [ww]")
! flip Z direction
      do 12 k=1,nlev_geos
          z = nlev_geos - k + 1
          ww(:,:,k,n) = var(:,:,z,1)
12    continue     
! T
      call check(nf_inq_varid(ncid,'T',varid),
     +       "getting varid for T")

      call check(nf_get_var(ncid,varid,var),
     +       " reading T")
! flip Z direction
      do 13 k=1,nlev_geos
          z = nlev_geos - k + 1
          tt(:,:,k,n) = var(:,:,z,1)
13    continue     
! QV specific humidity
      call check(nf_inq_varid(ncid,'QV',varid),
     +       "getting varid for QV")

      call check(nf_get_var(ncid,varid,var),
     +       " reading QV")    
! flip Z direction
      do 14 k=1,nlev_geos
          z = nlev_geos - k + 1
          qq(:,:,k,n) = var(:,:,z,1)
14    continue     
! PS
      call check(nf_inq_varid(ncid,'PS',varid),
     +       "getting varid for PS")

      call check(nf_get_var(ncid,varid,ps(:,:,1,n)),
     +       " reading PS")


! CHECK GRID SPECIFICATIONS
      gotGrid = 0
      call check(nf_inq_varid(ncid,'lon',varid),
     +       "getting varid for lon")
      call check(nf_get_var(ncid,varid,lon),
     +       "reading lon")
! Westernmost Longitude
      xaux = real(lon(1))

      call check(nf_inq_varid(ncid,'lat',varid),
     +       "getting varid for lat")
      call check(nf_get_var(ncid,varid,lat),
     +       "reading lat")
! Southernmost Latitude
      yaux = real(lat(1))

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
      gotGrid = 1
!
! CLOSING OF INPUT DATA FILE
!
      call check(nf_close(ncid),"closing" // trim(wfname(indj)))

!     error message if no fields found with correct first longitude in it
      if (gotGrid.eq.0) then
        print*,'***ERROR: input file has bad first lat/lon'
        stop
      endif

      if(levdiff2.eq.0) then
        iwmax=nlev_geos+1
        do 60 j=0,ny-1
          do 60 i=0,nx-1
60          ww(i,j,nlev_geos+1,n)=0.
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
      end
!;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
! NAME
!     check
! PURPOSE
!     tests the return value of an NF90 call
!     prints a message (loc) if the return value indicates an error
! INPUT
!     status : NF90 return value to be checked
!     loc    : use character string indicating where in the code the 
!              NF90 call is
! OUTPUT
!     Writes to the standard output the loc and the NF90 error
!  HISTORY
!     27 April P. Castellanos
!;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
      subroutine check(status, loc)
      use netcdf

      include'netcdf.inc'

      integer, intent(in) :: status
      character(len=*), intent(in) :: loc

      if(status /= NF_NOERR) then
        write (*,*) "Error at ", loc
        write (*,*) NF_STRERROR(status)
        stop 2
      end if

      end  
