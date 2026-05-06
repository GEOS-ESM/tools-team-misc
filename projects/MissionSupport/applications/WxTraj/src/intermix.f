      subroutine intermix(xt,yt,zt,zconst,indexf,itime,iter,ngrid,
     +uint,vint)
C                         i  i  i    i      i     i    i      i
C      o    o
********************************************************************************
*                                                                              *
*     Interpolation routine for mixing layer trajectories.                     *
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
* firstep            .true. for first iteration of petterssen                  *
* hmix               height of mixing layer in eta coordinates                 *
* idiff [s]          Temporal distance between the windfields used for interpol*
* induvz             index of the model layer beneath current position of traj.*
* init               .true. for first time step of trajectory                  *
* iter               number of iteration step                                  *
* itime [s]          current temporal position                                 *
* memtime(3) [s]     times of the wind fields in memory                        *
* pp [Pa]            pressure at mixing height                                 *
* ngrid              index which grid is to be used                            *

* nstop              =greater 0, if trajectory calculation is finished         *
* uint,vint [m/s]    interpolated wind components                              *
* xt,yt,zt           coordinates position for which wind data shall be calculat*
* zconst [m]         height of mixing layer                                    *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer i,itime,indexf,induvz,iter,ngrid
      real xt,yt,zt,zconst,uint,vint,uhelp,vhelp,qhelp
      real psint,xtn,ytn,hmix,weight,pp,ph1,ph2


C Determine nested grid coordinates
***********************************

      if (ngrid.gt.0) then
        xtn=(xt-xln(ngrid))*xresoln(ngrid)
        ytn=(yt-yln(ngrid))*yresoln(ngrid)
      endif


C Transformation from mixing height in [m] coordinate to height in eta coordinate.
C For this it is necessary to know the surface pressure.
**********************************************************************************

      if (ngrid.gt.0) then         ! nested grid
        call levlininterpoln(psn,maxnests,nxmaxn,nymaxn,1,ngrid,
     +  nxn,nyn,memind,xtn,ytn,1,memtime(indexf),memtime(indexf+1),
     +  itime,indexf,psint)
        hmix=zconst
        call etatrafo(xtn,ytn,hmix,memtime(indexf),memtime(indexf+1),
     +  itime,indexf,ngrid,psint)
      else                         ! mother grid
        call levlininterpol(ps,nxmax,nymax,1,nx,ny,memind,
     +  xt,yt,1,memtime(indexf),memtime(indexf+1),itime,indexf,psint)
        hmix=zconst
        call etatrafo(xt,yt,hmix,memtime(indexf),memtime(indexf+1),
     +  itime,indexf,ngrid,psint)
      endif


C Calculate the index of layers between which the mixed layer height is situated.
*********************************************************************************

      do 85 induvz=1,nuvz-1
        if (uvheight(induvz).ge.hmix) goto 86
85      continue
86    continue

C Mixed layer height is located between uvheight(induvz-1) and uvheight(induvz)
*******************************************************************************

C 1.) If mixed layer height is lower than the first model level
C     -> take wind data from first model level
***************************************************************

      if (induvz.eq.1) then
        zt=uvheight(1)
        if (ngrid.eq.0) then       ! mother grid
        if ((inpolkind.eq.1).and.(iter.ne.1)) then
          call levinterpol(uu,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,1,memtime(indexf),memtime(indexf+1),itime,
     +    indexf,uint)
          call levinterpol(vv,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,1,memtime(indexf),memtime(indexf+1),itime,
     +    indexf,vint)
        else                               ! linear interpolation
          call levlininterpol(uu,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,1,memtime(indexf),memtime(indexf+1),itime,
     +    indexf,uint)
          call levlininterpol(vv,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,1,memtime(indexf),memtime(indexf+1),itime,
     +    indexf,vint)
        endif

      else if (ngrid.gt.0) then      ! nested grid

        if ((inpolkind.eq.1).and.(iter.ne.1)) then
          call levinterpoln(uun,maxnests,nxmaxn,nymaxn,nuvzmax,ngrid,
     +    nxn,nyn,memind,xtn,ytn,1,memtime(indexf),memtime(indexf+1),
     +    itime,indexf,uint)
          call levinterpoln(vvn,maxnests,nxmaxn,nymaxn,nuvzmax,ngrid,
     +    nxn,nyn,memind,xtn,ytn,1,memtime(indexf),memtime(indexf+1),
     +    itime,indexf,vint)
        else                               ! linear interpolation
          call levlininterpoln(uun,maxnests,nxmaxn,nymaxn,nuvzmax,ngrid,
     +    nxn,nyn,memind,xtn,ytn,1,memtime(indexf),memtime(indexf+1),
     +    itime,indexf,uint)
          call levlininterpoln(vvn,maxnests,nxmaxn,nymaxn,nuvzmax,ngrid,
     +    nxn,nyn,memind,xtn,ytn,1,memtime(indexf),memtime(indexf+1),
     +    itime,indexf,vint)
        endif

        else                       ! polar stereographic grid
                                   ! only linear interpolation used
          call levlininterpol(uupol,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,1,memtime(indexf),memtime(indexf+1),itime,
     +    indexf,uint)
          call levlininterpol(vvpol,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,1,memtime(indexf),memtime(indexf+1),itime,
     +    indexf,vint)
        endif


C 2.) Normal case: mixed layer higher than the first model level
C     -> take average wind within mixed layer for trajectory displacement
*************************************************************************

      else
        weight=0.
        uhelp=0.
        vhelp=0.
        qhelp=0.
        ph1=psint

        do 140 i=1,induvz-2

C Interpolation of wind field data is done.
C -> Interpolated winds are weighted with the thicknesses of the layers
***********************************************************************

          if (ngrid.eq.0) then       ! mother grid
          if ((inpolkind.eq.1).and.(iter.ne.1)) then
            call levinterpol(uu,nxmax,nymax,nuvzmax,nx,ny,memind,
     +      xt,yt,i,memtime(indexf),memtime(indexf+1),itime,
     +      indexf,uint)
            call levinterpol(vv,nxmax,nymax,nuvzmax,nx,ny,memind,
     +      xt,yt,i,memtime(indexf),memtime(indexf+1),itime,
     +      indexf,vint)
          else                               ! bilinear interpolation
            call levlininterpol(uu,nxmax,nymax,nuvzmax,nx,ny,memind,
     +      xt,yt,i,memtime(indexf),memtime(indexf+1),itime,
     +      indexf,uint)
            call levlininterpol(vv,nxmax,nymax,nuvzmax,nx,ny,memind,
     +      xt,yt,i,memtime(indexf),memtime(indexf+1),itime,
     +      indexf,vint)
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

          else                       ! polar stereographic grid
                                     ! only linear interpolation used
            call levlininterpol(uupol,nxmax,nymax,nuvzmax,nx,ny,memind,
     +      xt,yt,i,memtime(indexf),memtime(indexf+1),itime,
     +      indexf,uint)
            call levlininterpol(vvpol,nxmax,nymax,nuvzmax,nx,ny,memind,
     +      xt,yt,i,memtime(indexf),memtime(indexf+1),itime,
     +      indexf,vint)
          endif
          ph2=akm(i+1)+bkm(i+1)*psint
          weight=weight+ph1-ph2
          uhelp=uhelp+uint*(ph1-ph2)
          vhelp=vhelp+vint*(ph1-ph2)
140       ph1=ph2


        if (ngrid.eq.0) then         ! mother grid
        if ((inpolkind.eq.1).and.(iter.ne.1)) then
          call levinterpol(uu,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,induvz-1,memtime(indexf),memtime(indexf+1),itime,
     +    indexf,uint)
          call levinterpol(vv,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,induvz-1,memtime(indexf),memtime(indexf+1),itime,
     +    indexf,vint)
        else                               ! bilinear interpolation
          call levlininterpol(uu,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,induvz-1,memtime(indexf),memtime(indexf+1),itime,
     +    indexf,uint)
          call levlininterpol(vv,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,induvz-1,memtime(indexf),memtime(indexf+1),itime,
     +    indexf,vint)
        endif

        else if (ngrid.gt.0) then      ! nested grid

          if ((inpolkind.eq.1).and.(iter.ne.1)) then
            call levinterpoln(uun,maxnests,nxmaxn,nymaxn,nuvzmax,ngrid,
     +      nxn,nyn,memind,xtn,ytn,induvz-1,memtime(indexf),
     +      memtime(indexf+1),itime,indexf,uint)
            call levinterpoln(vvn,maxnests,nxmaxn,nymaxn,nuvzmax,ngrid,
     +      nxn,nyn,memind,xtn,ytn,induvz-1,memtime(indexf),
     +      memtime(indexf+1),itime,indexf,vint)
          else                               ! linear interpolation
            call levlininterpoln(uun,maxnests,nxmaxn,nymaxn,nuvzmax,
     +      ngrid,nxn,nyn,memind,xtn,ytn,induvz-1,memtime(indexf),
     +      memtime(indexf+1),itime,indexf,uint)
            call levlininterpoln(vvn,maxnests,nxmaxn,nymaxn,nuvzmax,
     +      ngrid,nxn,nyn,memind,xtn,ytn,induvz-1,memtime(indexf),
     +      memtime(indexf+1),itime,indexf,vint)
          endif

        else                         ! polar stereographic grid
                                     ! only linear interpolation used
          call levlininterpol(uupol,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,induvz-1,memtime(indexf),memtime(indexf+1),itime,
     +    indexf,uint)
          call levlininterpol(vvpol,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,induvz-1,memtime(indexf),memtime(indexf+1),itime,
     +    indexf,vint)
        endif
        ph2=pp(psint,hmix)
        weight=weight+ph1-ph2
        uhelp=uhelp+uint*(ph1-ph2)
        vhelp=vhelp+vint*(ph1-ph2)

        uint=uhelp/weight
        vint=vhelp/weight
      endif
      zt=hmix

      return
      end
