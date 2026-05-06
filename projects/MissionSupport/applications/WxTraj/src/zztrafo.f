      subroutine zztrafo(ngrid,xt,yt,zt,itime1,itime2,itime,indexf,
     +psint,ht)
***********************************************************************
*                                                                     *
*             TRAJECTORY MODEL SUBROUTINE ZZTRAFO                     *
*                                                                     *
***********************************************************************
*                                                                     *
*             AUTHOR:      G. WOTAWA                                  *
*             DATE:        1994-04-07                                 *
*             LAST UPDATE: 1999-01-07 Adaptation to nesting           *
*                                                                     *
***********************************************************************
*                                                                     *
* DESCRIPTION: This subroutine transforms the vertical coordinate     *
*              eta (ECMWF) to geometric height [m] above model        *
*              orography (relative height)                            *
*              Method: vertical integration of hydrostatic equation   *
*                                                                     *
* REMARK: For the calculation of geometric height, the virtual        *
*         temperature difference has been neglected. The gas constant *
*         for dry air has been used.                                  *
*                                                                     *
***********************************************************************
*                                                                     *
* INPUT:                                                              *
*                                                                     *
* ngrid                      number of nesting level to be used       *
* xt                         x-coordinate of point [grid units]       *
* yt                         y-coordinate of point [grid units]       *
* zt                         z-coordinate of point [eta (ECMWF)]      *
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
* ht                         geometric height [m]                     *
*                                                                     *
***********************************************************************
*
      include 'includepar'
      include 'includecom'

      integer itime1,itime2,itime,indexf,i,k,ngrid
      real xt,yt,zt,ht,psint,fract,pp1,pp2,tv
  
      real tlev(nuvzmax)
      real zzlev1,zzlev2

      do 10 k=2,nwz
         if(zt.le.wheight(k)) goto 20
10    continue
      k=nwz
20    fract=(zt-wheight(k-1))/(wheight(k)-wheight(k-1))
*
* calculate interpolated vertical temperature profile on model levels
*
      do 30 i=1,k-1
        if (ngrid.gt.0) then
          call levlininterpoln(ttn,maxnests,nxmaxn,nymaxn,nuvzmax,ngrid,
     +    nxn,nyn,memind,xt,yt,i,itime1,itime2,itime,indexf,tlev(i))
        else
          call levlininterpol(tt,nxmax,nymax,nuvzmax,nx,ny,memind,
     +    xt,yt,i,itime1,itime2,itime,indexf,tlev(i))
        endif
30      continue
*
* calculate layer indices between which zt is situated
*
      zzlev1=0
      do 40 i=2,k-1
        pp1=akm(i-1)+bkm(i-1)*psint
        pp2=akm(i)+bkm(i)*psint
        tv=tlev(i-1)
40      zzlev1=zzlev1+r_air/ga*log(pp1/pp2)*tv
      pp1=akm(k-1)+bkm(k-1)*psint
      pp2=akm(k)  +bkm(k)*psint
      tv=tlev(k-1)
      zzlev2=zzlev1+r_air/ga*log(pp1/pp2)*tv

      ht=zzlev1*(1.-fract)+zzlev2*fract
      return
      end
