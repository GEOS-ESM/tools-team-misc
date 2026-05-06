      subroutine inter3d(xt,yt,zt,indexf,itime,init,firststep,
     +iter,lkindz,ngrid,uint,vint,dzdt,indwz)
C                        i  i  i    i     i    i       i
C      i     i      i    o    o    o     o
********************************************************************************
*                                                                              *
*     Interpolation routine for 3-dimensional trajectories.                    *
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
* dzdt               wind components in grid units per second                  *
* firstep            .true. for first iteration of petterssen                  *
* idiff [s]          Temporal distance between the windfields used for interpol*
* indwz              index of the model layer beneath current position of traj.*
* init               .true. for first time step of trajectory                  *
* iter               number of time iteration step                             *
* itime [s]          current temporal position                                 *
* memtime(3) [s]     times for the wind fields in memory                       *
* ngrid              index which grid is to be used                            *
* nstop              =greater 0, if trajectory calculation is finished         *
* uint,vint,wint [m/s] interpolated wind components                            *
* xt,yt,zt           coordinates position for which wind data shall be calculat*
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer i,itime,indexf,indwz,iter,lkindz,ngrid
      real xt,yt,zt,dzdt,uint,vint,wint,psint,eta,xtn,ytn
      logical firststep,init


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


C For a new trajectory and the first iteration of petterssen:
C Transformation from height in [m] coordinate to height in eta coordinate.
C Or: transformation from height in [Pa] to height in eta coordinate.
***************************************************************************

      if (init.and.firststep) then
        if ((lkindz.eq.1).or.(lkindz.eq.2)) then
          if (ngrid.gt.0) then       ! nested grid
            call etatrafo(xtn,ytn,zt,memtime(indexf),memtime(indexf+1),
     +      itime,indexf,ngrid,psint)
          else                       ! mother grid
            call etatrafo(xt,yt,zt,memtime(indexf),memtime(indexf+1),
     +      itime,indexf,ngrid,psint)
          endif
        else if (lkindz.eq.3) then
          zt=eta(psint,zt)
        endif
      endif


C Calculate index of (w) layers between the trajectory point is situated.
C Trajectory point is located between wheight(indwz) and wheight(indwz+1)
*************************************************************************

      do 95 i=2,nwz
        if (wheight(i).ge.zt) goto 96
95      continue
96    indwz=i-1



C Interpolation of wind field data is done.
C Either linear interpolation (if selected and for first iteration
C of petterssen) or bicubic interpolation
*******************************************

      if (ngrid.eq.0) then           ! mother grid
        if ((inpolkind.eq.1).and.(iter.ne.1)) then
        call interpol(uu,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,uvheight,
     +  xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,indexf,uint)
        call interpol(vv,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,uvheight,
     +  xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,indexf,vint)
        call interpol(ww,nxmax,nymax,nwzmax,nx,ny,nwz,memind,wheight,
     +  xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,indexf,wint)
        else                               ! bilinear interpolation
        call lininterpol(uu,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,
     +  uvheight,xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,
     +  indexf,uint)
        call lininterpol(vv,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,
     +  uvheight,xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,
     +  indexf,vint)
        call lininterpol(ww,nxmax,nymax,nwzmax,nx,ny,nwz,memind,
     +  wheight,xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,
     +  indexf,wint)
        endif

      else if (ngrid.gt.0) then      ! nested grid

        if ((inpolkind.eq.1).and.(iter.ne.1)) then
        call interpoln(uun,maxnests,nxmaxn,nymaxn,nuvzmax,
     +  ngrid,nxn,nyn,nuvz,memind,uvheight,xtn,ytn,zt,memtime(indexf),
     +  memtime(indexf+1),itime,indexf,uint)
        call interpoln(vvn,maxnests,nxmaxn,nymaxn,nuvzmax,
     +  ngrid,nxn,nyn,nuvz,memind,uvheight,xtn,ytn,zt,memtime(indexf),
     +  memtime(indexf+1),itime,indexf,vint)
        call interpoln(wwn,maxnests,nxmaxn,nymaxn,nwzmax,
     +  ngrid,nxn,nyn,nwz,memind,wheight,xtn,ytn,zt,memtime(indexf),
     +  memtime(indexf+1),itime,indexf,wint)

        else                               ! bilinear interpolation
        call lininterpoln(uun,maxnests,nxmaxn,nymaxn,nuvzmax,
     +  ngrid,nxn,nyn,nuvz,memind,uvheight,xtn,ytn,zt,memtime(indexf),
     +  memtime(indexf+1),itime,indexf,uint)
        call lininterpoln(vvn,maxnests,nxmaxn,nymaxn,nuvzmax,
     +  ngrid,nxn,nyn,nuvz,memind,uvheight,xtn,ytn,zt,memtime(indexf),
     +  memtime(indexf+1),itime,indexf,vint)
        call lininterpoln(wwn,maxnests,nxmaxn,nymaxn,nwzmax,
     +  ngrid,nxn,nyn,nwz,memind,wheight,xtn,ytn,zt,memtime(indexf),
     +  memtime(indexf+1),itime,indexf,wint)
        endif

      else                           ! polar stereographic grid
                                     ! only linear interpolation used

        call lininterpol(uupol,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,
     +  uvheight,xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,
     +  indexf,uint)
        call lininterpol(vvpol,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,
     +  uvheight,xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,
     +  indexf,vint)
        call lininterpol(ww,nxmax,nymax,nwzmax,nx,ny,nwz,memind,
     +  wheight,xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,
     +  indexf,wint)
      endif


      call wtransform(wint,psint,zt,indwz,dzdt)
      
      return
      end
