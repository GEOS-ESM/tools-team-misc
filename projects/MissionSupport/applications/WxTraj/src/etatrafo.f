      subroutine etatrafo(xt,yt,zt,itime1,itime2,itime,indexf,ngrid,
     +psint)
***********************************************************************
*                                                                     *
*             TRAJECTORY MODEL SUBROUTINE ETATRAFO                    *
*                                                                     *
***********************************************************************
*                                                                     *
*             AUTHOR:      G. WOTAWA                                  *
*             DATE:        1994-04-06                                 *
*             LAST UPDATE: ----------                                 *
*                                                                     *
***********************************************************************
*                                                                     *
* DESCRIPTION: This subroutine transforms the vertical coordinate     *
*              from z-coordinate system [m] to eta-coordinate         *
*              system [ECMWF]                                         *
*              Remark: Just call after initialization of a new        *
*              trajectory (first call of getwind)                     *
*                                                                     *
***********************************************************************
*                                                                     *
* INPUT:                                                              *
*                                                                     *
* xt                         x-coordinate of point [grid units]       *
* yt                         y-coordinate of point [grid units]       *
* zt                         z-coordinate of point [m]                *
* itime1                     time [s] of first windfield              *
* itime2                     time [s] of second windfield             *
* itime                      time [s] of calculation                  *
* indexf                     time index of field xx                   *
* psint                      surface pressure at point (xt,yt) [Pa]   *
*                                                                     *
***********************************************************************
*                                                                     *
* OUTPUT:                                                             *
*                                                                     *
* zt                         z-coordinate of point [eta ECMWF]        *
*                                                                     *
***********************************************************************
*
      include 'includepar'
      include 'includecom'

      integer itime1,itime2,itime,indexf,k,ngrid
      real xt,yt,zt,psint,fract,pp1,pp2,tv
      real tlev(nuvzmax),zzlev(nwzmax)
*
* calculate interpolated vertical temperature profile on model levels
*
      do 10 k=1,nuvz
       if (ngrid.gt.0) then
         call levlininterpoln(ttn,maxnests,nxmaxn,nymaxn,nuvzmax,ngrid,
     +   nxn,nyn,memind,xt,yt,k,itime1,itime2,itime,indexf,tlev(k))
       else
         call levlininterpol(tt,nxmax,nymax,nuvzmax,nx,ny,memind,
     +   xt,yt,k,itime1,itime2,itime,indexf,tlev(k))
       endif
10     continue
*
* calculate geometric height on model levels
*
      zzlev(1)=0.
      if (zt.lt.zzlev(1)) then
        write(*,*) ' TRAJECTORY MODEL: NOTICE - STARTING POINT OUT'//
     &             ' OF MODEL DOMAIN'
        write(*,*) ' (VERTICAL) HAS BEEN DETECTED --> IT IS SET TO'//
     &             ' THE BOTTOM OF THE MODEL ...'
        zt=uvheight(1)
        return
      endif
      do 15 k=2,nwz
        pp1=akm(k-1)+bkm(k-1)*psint
        pp2=akm(k)+bkm(k)*psint
        tv=tlev(k-1)      !! NO HUMIDITY INFORMATION AVAILABLE IN FLEXTRA
        zzlev(k)=zzlev(k-1)+r_air/ga*log(pp1/pp2)*tv
        if(zt.lt.zzlev(k)) goto 20
15    continue
      write(*,*) ' TRAJECTORY MODEL: NOTICE - STARTING POINT OUT'//
     &           ' OF MODEL DOMAIN'
      write(*,*) ' (VERTICAL) HAS BEEN DETECTED --> IT IS SET TO'//
     &           ' THE TOP OF THE MODEL ...'
      zt=uvheight(nuvz)
      return
20    fract=(zt-zzlev(k-1))/(zzlev(k)-zzlev(k-1))
      zt=wheight(k-1)*(1.-fract)+wheight(k)*fract
      if(zt.lt.uvheight(1)) zt=uvheight(1)
      if(zt.gt.uvheight(nuvz)) zt=uvheight(nuvz)
      return
      end
