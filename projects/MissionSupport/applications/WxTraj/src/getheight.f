      subroutine getheight(itime,xt,yt,zt,pt,ht)
***********************************************************************
*                                                                     * 
*             TRAJECTORY MODEL SUBROUTINE GETHEIGHT                   *
*                                                                     *
***********************************************************************
*                                                                     * 
*             AUTHOR:      G. WOTAWA                                  *
*             DATE:        1994-04-07                                 *
*             Adaptation to Nesting: A. Stohl, 1999-01-07             * 
*             More efficient determination of wind field index,       * 
*             A. Stohl, 2000-01-30                                    * 
***********************************************************************
*                                                                     *
* DESCRIPTION: This subroutine calculates pressure [Pa] and geometric *
*              height [m] along trajectory for a given vertical       *
*              coordinate zt (eta ECMWF) on a given position (xt,yt). *
*                                                                     *
***********************************************************************
*                                                                     *
* INPUT:                                                              *
*                                                                     *
* itime     time relative to beginning of calculation period          *
* xt        x coordinate of point [grid units]                        *
* yt        y coordinate of point [grid units]                        *
* zt        z coordinate of point [units]                             *
*                                                                     *
***********************************************************************
*                                                                     *
* OUTPUT:                                                             *
*                                                                     *
* pt        pressure [Pa]                                             *
* ht        geometric height [m]                                      *
*                                                                     *
***********************************************************************
*
      include 'includepar'
      include 'includecom'

      integer itime,indexf,j,ngrid

      real pp,xt,yt,zt,pt,ht,psint,xtn,ytn


C  Determine the times of the wind fields needed for interpolation
******************************************************************

      if ((ldirect*memtime(1).lt.ldirect*itime).and.
     +(ldirect*memtime(2).ge.ldirect*itime)) then ! between 1st and 2nd
        indexf=1
      else if((ldirect*memtime(2).lt.ldirect*itime).and.
     +(ldirect*memtime(3).ge.ldirect*itime)) then ! between 2nd and 3rd
        indexf=2
      endif


C Determine which nesting level to be used
******************************************

      ngrid=0
      do 12 j=numbnests,1,-1
        if ((xt.gt.xln(j)).and.(xt.lt.xrn(j)).and.
     +  (yt.gt.yln(j)).and.(yt.lt.yrn(j))) then
          ngrid=j
          goto 13
        endif
12      continue
13    continue

      if (ngrid.eq.0) then
        xtn=xt
        ytn=yt
        call levlininterpol(ps,nxmax,nymax,1,nx,ny,memind,
     +  xt,yt,1,memtime(indexf),memtime(indexf+1),itime,indexf,psint)
      else
        xtn=(xt-xln(ngrid))*xresoln(ngrid)
        ytn=(yt-yln(ngrid))*yresoln(ngrid)
        call levlininterpoln(psn,maxnests,nxmaxn,nymaxn,1,ngrid,
     +  nxn,nyn,memind,xtn,ytn,1,memtime(indexf),memtime(indexf+1),
     +  itime,indexf,psint)
      endif

      pt=pp(psint,zt)

      call zztrafo(ngrid,xtn,ytn,zt,memtime(indexf),memtime(indexf+1),
     +itime,indexf,psint,ht)

      return
      end
