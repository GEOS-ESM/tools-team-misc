      subroutine interisentrop(xt,yt,zt,thetaconst,indexf,itime,init,
     +firststep,iter,lkindz,ngrid,uint,vint)
C                              i  i  i/o    i/o      i     i    i
C         i      i     i      i    o    o
********************************************************************************
*                                                                              *
*     Interpolation routine for isentropic trajectories.                       *
*                                                                              *
* The user gives the vertical coordinate in metres. In the first time step of  *
* the trajectory, metres are transformed to Kelvin. In all time steps these    *
* Kelvin are kept constant and transformed onward to the eta coordinate.       *
*                                                                              *
*                                                                              *
*     Author: A. Stohl                                                         *
*                                                                              *
*     27 April 1994                                                            *
*                                                                              *
*     Update: 23 December 1998 (Use of global domain and nesting)              *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* firststep          .true. for first step of petterssen, .false. for iteration*
* idiff [s]          Temporal distance between the windfields used for interpol*
* indz1,indz2        indices of boundary of unstable region                    *
* init               .true. for first time step of trajectory, .false. for othe*
* iter               number of iteration step                                  *
* itime [s]          current temporal position                                 *
* lkindz             unit of z coordinate (1:masl, 2:magl, 3:hPa)              *
* memtime(3) [s]     times of the wind fields in memory                        *
* ngrid              index which grid is to be used                            *
* nstop              =greater 0, if trajectory calculation is finished         *
* psint [Pa]         interpolated surface pressure                             *
* temp [K]           temperature at trajectory position                        *
* thetaconst [Pa]    theta level, for which trajectories shall be calculated   *
* uint,vint [m/s]    interpolated wind components                              *
* uvheight           heights in which u,v and t are given                      *
* xt,yt,zt           coordinates position for which wind data shall be calculat*
*                                                                              *
* Constants:                                                                   *
* kappa              exponent for calculating potential temperature            *
*                                                                              *
* Function:                                                                    *
* pp                 calculates the pressure at a given eta level              *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer itime,indexf,iter,indz1,indz2,i,lkindz,ngrid
      real xt,yt,zt,uint,vint,psint,thetaconst,phelp,pp,xtn,ytn
      real weight,uhelp,vhelp,qhelp,ph1,ph2,eta
      logical init,firststep

      indz1=0
      indz2=0

C Determine nested grid coordinates
***********************************

      if (ngrid.gt.0) then
        xtn=(xt-xln(ngrid))*xresoln(ngrid)
        ytn=(yt-yln(ngrid))*yresoln(ngrid)
      endif

C Calculate the surface pressure.
*********************************

      if (ngrid.gt.0) then         ! nested grid
        call levlininterpoln(psn,maxnests,nxmaxn,nymaxn,1,ngrid,nxn,nyn,
     +  memind,xtn,ytn,1,memtime(indexf),memtime(indexf+1),itime,
     +  indexf,psint)
      else                         ! mother grid
        call levlininterpol(ps,nxmax,nymax,1,nx,ny,memind,xt,yt,
     +  1,memtime(indexf),memtime(indexf+1),itime,indexf,psint)
      endif


******************************************************************************
C For a new trajectory and the first iteration of petterssen:
C thetaconst is given in [m]
C 1. Transformation from height in [m] coordinate to height in eta coordinate.
C    Or: Transformation from hPa to eta.
C 2. Transformation from height in eta to height in theta [K] coordinate.
******************************************************************************

      if (init.and.firststep) then
        if ((lkindz.eq.1).or.(lkindz.eq.2)) then
          if (ngrid.gt.0) then       ! nested grid
            call etatrafo(xtn,ytn,thetaconst,memtime(indexf),
     +      memtime(indexf+1),itime,indexf,ngrid,psint)
          else                       ! mother grid
            call etatrafo(xt,yt,thetaconst,memtime(indexf),
     +      memtime(indexf+1),itime,indexf,ngrid,psint)
          endif
          zt=thetaconst              ! now thetaconst is given in eta
        else if (lkindz.eq.3) then
          zt=eta(psint,zt)
        endif

C Calculate pressure and potential temperature for current position
*******************************************************************

        phelp=pp(psint,zt)
        if (ngrid.gt.0) then       ! nested grid
          call lininterpoln(thn,maxnests,nxmaxn,nymaxn,nuvzmax,
     +    ngrid,nxn,nyn,nuvz,memind,uvheight,xtn,ytn,zt,memtime(indexf),
     +    memtime(indexf+1),itime,indexf,thetaconst)
        else                       ! mother grid
          call lininterpol(th,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,
     +    uvheight,xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,
     +    indexf,thetaconst)
        endif
      else

********************************************************************************
C For all other time steps:
C Transformation from height in theta [K] coordinate to height in eta coordinate
********************************************************************************

        if (ngrid.gt.0) then       ! nested grid
          call geteta(xtn,ytn,zt,thetaconst,memtime(indexf),
     +    memtime(indexf+1),itime,indexf,psint,ngrid,indz1,indz2)
        else                       ! mother grid
          call geteta(xt,yt,zt,thetaconst,memtime(indexf),
     +    memtime(indexf+1),itime,indexf,psint,ngrid,indz1,indz2)
        endif
      endif


C Interpolation of wind field data is done.
*****************************************************************************

      if (indz1.eq.indz2) then         ! stable level
      if (ngrid.eq.0) then          ! mother grid
      if ((inpolkind.eq.1).and.(iter.ne.1)) then
        call interpol(uu,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,uvheight,
     +  xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,indexf,uint)
        call interpol(vv,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,uvheight,
     +  xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,indexf,vint)
      else                               ! bilinear interpolation
        call lininterpol(uu,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,
     +  uvheight,xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,
     +  indexf,uint)
        call lininterpol(vv,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,
     +  uvheight,xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,
     +  indexf,vint)
      endif

      else if (ngrid.gt.0) then      ! nested grid

        if ((inpolkind.eq.1).and.(iter.ne.1)) then
        call interpoln(uun,maxnests,nxmaxn,nymaxn,nuvzmax,
     +  ngrid,nxn,nyn,nuvz,memind,uvheight,xtn,ytn,zt,memtime(indexf),
     +  memtime(indexf+1),itime,indexf,uint)
        call interpoln(vvn,maxnests,nxmaxn,nymaxn,nuvzmax,
     +  ngrid,nxn,nyn,nuvz,memind,uvheight,xtn,ytn,zt,memtime(indexf),
     +  memtime(indexf+1),itime,indexf,vint)
        else                               ! bilinear interpolation
        call lininterpoln(uun,maxnests,nxmaxn,nymaxn,nuvzmax,
     +  ngrid,nxn,nyn,nuvz,memind,uvheight,xtn,ytn,zt,memtime(indexf),
     +  memtime(indexf+1),itime,indexf,uint)
        call lininterpoln(vvn,maxnests,nxmaxn,nymaxn,nuvzmax,
     +  ngrid,nxn,nyn,nuvz,memind,uvheight,xtn,ytn,zt,memtime(indexf),
     +  memtime(indexf+1),itime,indexf,vint)
        endif

      else                          ! polar stereographic grid
                                    ! only linear interpolation used
        call lininterpol(uupol,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,
     +  uvheight,xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,
     +  indexf,uint)
        call lininterpol(vvpol,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,
     +  uvheight,xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,
     +  indexf,vint)
      endif

      else                             ! unstable layer

C -> take the layer averaged wind velocity (weighted with pressure increments)
******************************************************************************

        weight=0.
        uhelp=0.
        vhelp=0.
        qhelp=0.
        ph1=akz(indz1)+bkz(indz1)*psint

        do 10 i=indz1,indz2-1
          if (ngrid.ge.0) then          ! mother grid
          if ((inpolkind.eq.1).and.(iter.ne.1)) then
            call levinterpol(uu,nxmax,nymax,nuvzmax,nx,ny,memind,
     +      xt,yt,i,memtime(indexf),memtime(indexf+1),itime,indexf,
     +      uint)
            call levinterpol(vv,nxmax,nymax,nuvzmax,nx,ny,memind,
     +      xt,yt,i,memtime(indexf),memtime(indexf+1),itime,indexf,
     +      vint)
          else                               ! bilinear interpolation
            call levlininterpol(uu,nxmax,nymax,nuvzmax,nx,ny,memind,
     +      xt,yt,i,memtime(indexf),memtime(indexf+1),itime,indexf,
     +      uint)
            call levlininterpol(vv,nxmax,nymax,nuvzmax,nx,ny,memind,
     +      xt,yt,i,memtime(indexf),memtime(indexf+1),itime,indexf,
     +      vint)
          endif

          else if (ngrid.gt.0) then      ! nested grid

            if ((inpolkind.eq.1).and.(iter.ne.1)) then
          call levinterpoln(uun,maxnests,nxmaxn,nymaxn,nuvzmax,ngrid,
     +    nxn,nyn,memind,xtn,ytn,i,memtime(indexf),memtime(indexf+1),
     +    itime,indexf,uint)
          call levinterpoln(vvn,maxnests,nxmaxn,nymaxn,nuvzmax,ngrid,
     +    nxn,nyn,memind,xtn,ytn,i,memtime(indexf),memtime(indexf+1),
     +    itime,indexf,vint)
            else                               ! linear interpolation
          call levlininterpoln(uun,maxnests,nxmaxn,nymaxn,nuvzmax,ngrid,
     +    nxn,nyn,memind,xtn,ytn,i,memtime(indexf),memtime(indexf+1),
     +    itime,indexf,uint)
          call levlininterpoln(vvn,maxnests,nxmaxn,nymaxn,nuvzmax,ngrid,
     +    nxn,nyn,memind,xtn,ytn,i,memtime(indexf),memtime(indexf+1),
     +    itime,indexf,vint)
            endif

          else                          ! polar stereographic grid
                                        ! only linear interpolation used
            call levlininterpol(uupol,nxmax,nymax,nuvzmax,nx,ny,memind,
     +      xt,yt,i,memtime(indexf),memtime(indexf+1),itime,indexf,
     +      uint)
            call levlininterpol(vvpol,nxmax,nymax,nuvzmax,nx,ny,memind,
     +      xt,yt,i,memtime(indexf),memtime(indexf+1),itime,indexf,
     +      vint)
          endif
          ph2=akm(i+1)+bkm(i+1)*psint
          weight=weight+ph1-ph2
          uhelp=uhelp+uint*(ph1-ph2)
          vhelp=vhelp+vint*(ph1-ph2)
10        ph1=ph2

        if (ngrid.ge.0) then          ! mother grid
        if ((inpolkind.eq.1).and.(iter.ne.1)) then
          call levinterpol(uu,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,indz2,memtime(indexf),memtime(indexf+1),itime,indexf,
     +    uint)
          call levinterpol(vv,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,indz2,memtime(indexf),memtime(indexf+1),itime,indexf,
     +    vint)
        else                               ! bilinear interpolation
          call levlininterpol(uu,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,indz2,memtime(indexf),memtime(indexf+1),itime,indexf,
     +    uint)
          call levlininterpol(vv,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,indz2,memtime(indexf),memtime(indexf+1),itime,indexf,
     +    vint)
        endif

        else if (ngrid.gt.0) then      ! nested grid

          if ((inpolkind.eq.1).and.(iter.ne.1)) then
          call levinterpoln(uun,maxnests,nxmaxn,nymaxn,nuvzmax,ngrid,
     +    nxn,nyn,memind,xtn,ytn,indz2,memtime(indexf),
     +    memtime(indexf+1),itime,indexf,uint)
          call levinterpoln(vvn,maxnests,nxmaxn,nymaxn,nuvzmax,ngrid,
     +    nxn,nyn,memind,xtn,ytn,indz2,memtime(indexf),
     +    memtime(indexf+1),itime,indexf,vint)
          else                               ! linear interpolation
          call levlininterpoln(uun,maxnests,nxmaxn,nymaxn,nuvzmax,ngrid,
     +    nxn,nyn,memind,xtn,ytn,indz2,memtime(indexf),
     +    memtime(indexf+1),itime,indexf,uint)
          call levlininterpoln(vvn,maxnests,nxmaxn,nymaxn,nuvzmax,ngrid,
     +    nxn,nyn,memind,xtn,ytn,indz2,memtime(indexf),
     +    memtime(indexf+1),itime,indexf,vint)
          endif

        else                          ! polar stereographic grid
          call levlininterpol(uupol,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,indz2,memtime(indexf),memtime(indexf+1),itime,indexf,
     +    uint)
          call levlininterpol(vvpol,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,indz2,memtime(indexf),memtime(indexf+1),itime,indexf,
     +    vint)
        endif

        ph2=akz(indz2)+bkz(indz2)*psint
        weight=weight+ph1-ph2
        uhelp=uhelp+uint*(ph1-ph2)
        vhelp=vhelp+vint*(ph1-ph2)

        uint=uhelp/weight
        vint=vhelp/weight
      endif


      return
      end
