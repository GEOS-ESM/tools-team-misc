      subroutine getmet(itime,xt,yt,zt,qqint,pvint,thint)
C                         i   i  i  i    i     i     i
********************************************************************************
*                                                                              *
*     Interpolation of meteorological data (i.e. specific humidity, potential  *
*     vorticity and potential temperature) onto trajectory positions.          *
*                                                                              *
*     Author: A. Stohl                                                         *
*                                                                              *
*     29 January 2000                                                          *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* itime [s]          current temporal position                                 *
* xt,yt,zt           coordinates of position for which data shall be interpolat*
* pvint              interpolated potential vorticity                          *
* qqint              interpolated specific humidity                            *
* thint              interpolated potential temperature                        *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer itime,indexf,ngrid,j
      real xt,yt,zt,xtn,ytn,qqint,pvint,thint


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


C Do linear interpolation of the meteo data
*******************************************

      if (ngrid.eq.0) then         ! mother grid
        call lininterpol(qq,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,
     +  uvheight,xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,
     +  indexf,qqint)
        call lininterpol(pv,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,
     +  uvheight,xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,
     +  indexf,pvint)
        call lininterpol(th,nxmax,nymax,nuvzmax,nx,ny,nuvz,memind,
     +  uvheight,xt,yt,zt,memtime(indexf),memtime(indexf+1),itime,
     +  indexf,thint)

      else if (ngrid.gt.0) then      ! nested grid

C Determine nested grid coordinates
***********************************

        xtn=(xt-xln(ngrid))*xresoln(ngrid)
        ytn=(yt-yln(ngrid))*yresoln(ngrid)

        call lininterpoln(qqn,maxnests,nxmaxn,nymaxn,nuvzmax,
     +  ngrid,nxn,nyn,nuvz,memind,uvheight,xtn,ytn,zt,memtime(indexf),
     +  memtime(indexf+1),itime,indexf,qqint)
        call lininterpoln(pvn,maxnests,nxmaxn,nymaxn,nuvzmax,
     +  ngrid,nxn,nyn,nuvz,memind,uvheight,xtn,ytn,zt,memtime(indexf),
     +  memtime(indexf+1),itime,indexf,pvint)
        call lininterpoln(thn,maxnests,nxmaxn,nymaxn,nuvzmax,
     +  ngrid,nxn,nyn,nuvz,memind,uvheight,xtn,ytn,zt,memtime(indexf),
     +  memtime(indexf+1),itime,indexf,thint)

      endif

      return
      end
