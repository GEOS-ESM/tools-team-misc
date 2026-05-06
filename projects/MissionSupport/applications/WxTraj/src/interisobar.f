      subroutine interisobar(xt,yt,zt,pconst,indexf,itime,init,
     +firststep,iter,lkindz,ngrid,uint,vint)
C                            i  i  i/o  i      i      i    i
C         i      i     i      i    o    o
********************************************************************************
*                                                                              *
*     Interpolation routine for isobaric trajectories.                         *
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
* eta                eta coordinate of pressure level                          *
* firststep          .true. for first iteration of petterssen                  *
* idiff [s]          Temporal distance between the windfields used for interpol*
* init               .true. for first time step of trajectory                  *
* iter               number of iteration step                                  *
* itime [s]          current temporal position                                 *
* lkindz             unit of z coordinate (1:masl, 2:magl, 3:hPa)              *
* memtime(3) [s]     times of the wind fields in memory                        *
* ngrid              index which grid is to be used                            *

* nstop              =greater 0, if trajectory calculation is finished         *
* pconst [Pa]        pressure level, for which trajectories shall be calculated*
* uint,vint [m/s]    interpolated wind components                              *
* xt,yt,zt           coordinates position for which wind data shall be calculat*
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer itime,indexf,iter,lkindz,ngrid
      real xt,yt,zt,uint,vint,psint,eta,pconst,pp,xtn,ytn
      logical init,firststep


C Determine nested grid coordinates
***********************************

      if (ngrid.gt.0) then
        xtn=(xt-xln(ngrid))*xresoln(ngrid)
        ytn=(yt-yln(ngrid))*yresoln(ngrid)
      endif


C For a new trajectory and the first iteration of petterssen:
C Transformation from height in [m] coordinate to height in eta coordinate.
C Or: Transformation from hPa to eta.
C For this it is necessary to know the surface pressure.
***************************************************************************

      if (init.and.firststep) then
C Calculate the surface pressure.
*********************************

        if (ngrid.gt.0) then        ! nested grid
          call levlininterpoln(psn,maxnests,nxmaxn,nymaxn,1,ngrid,
     +    nxn,nyn,memind,xtn,ytn,1,memtime(indexf),memtime(indexf+1),
     +    itime,indexf,psint)
        else                        ! mother grid
          call levlininterpol(ps,nxmax,nymax,1,nx,ny,memind,xt,yt,
     +    1,memtime(indexf),memtime(indexf+1),itime,indexf,psint)
        endif

        if ((lkindz.eq.1).or.(lkindz.eq.2)) then
          if (ngrid.gt.0) then      ! nested grid
            call etatrafo(xtn,ytn,zt,memtime(indexf),memtime(indexf+1),
     +      itime,indexf,ngrid,psint)
          else                      ! mother grid
            call etatrafo(xt,yt,zt,memtime(indexf),memtime(indexf+1),
     +      itime,indexf,ngrid,psint)
          endif
          pconst=pp(psint,zt)
        else if (lkindz.eq.3) then
          zt=eta(psint,zt)
        endif
      endif


C Transformation from height in pressure coordinate to height in eta coordinate.
C For this it is necessary to know the surface pressure.
*********************************************************************************

      if (ngrid.gt.0) then
        call levlininterpoln(psn,maxnests,nxmaxn,nymaxn,1,ngrid,
     +  nxn,nyn,memind,xtn,ytn,1,memtime(indexf),memtime(indexf+1),
     +  itime,indexf,psint)
      else
        call levlininterpol(ps,nxmax,nymax,1,nx,ny,memind,xt,yt,
     +  1,memtime(indexf),memtime(indexf+1),itime,indexf,psint)
      endif
      zt=eta(psint,pconst)


C Interpolation of wind field data is done.
*****************************************************************************

      if (ngrid.eq.0) then            ! mother grid
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

      else                            ! polar stereographic grid

        call lininterpol(uupol,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,
     +  uvheight,xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,
     +  indexf,uint)
        call lininterpol(vvpol,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,
     +  uvheight,xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,
     +  indexf,vint)
      endif

      return
      end
