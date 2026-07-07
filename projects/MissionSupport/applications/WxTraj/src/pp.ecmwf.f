      real function pp(psurf,eta)
***********************************************************************
*                                                                     *
*             TRAJECTORY MODEL SUBROUTINE PP                          *
*                                                                     *
***********************************************************************
*                                                                     *
*             AUTHOR:      G. WOTAWA                                  *
*             DATE:        1994-04-07                                 *
*             LAST UPDATE: ----------                                 *
*                                                                     *
***********************************************************************
*                                                                     *
* DESCRIPTION: This function computes the pressure as a function of   *
*              eta (ECMWF) and surface pressure. The interpolation    *
*              between the nearest model layers is performed linear   *
*                                                                     *
***********************************************************************
*                                                                     *
* INPUT:                                                              *
*                                                                     *
* psurf        surface pressure [Pa]                                  *
* eta          vertical coordinate eta (ECMWF)                        *
*                                                                     *
***********************************************************************
*                                                                     *
* OUTPUT:                                                             *
*                                                                     *
* pp           pressure [Pa]                                          *
*                                                                     *
***********************************************************************
*
      include 'includepar'
      include 'includecom'

      integer k
      real psurf,eta,fract,pp1,pp2

*
* SEE BETWEEN WHICH MODEL LAYERS ETA IS SITUATED
*
      do 10 k=2,nwz

         if(wheight(k).gt.eta) goto 20

10    continue

      k=nwz

20    fract=(eta-wheight(k-1))/(wheight(k)-wheight(k-1))

      pp1=akm(k-1)+bkm(k-1)*psurf
      pp2=akm(k)+bkm(k)*psurf

      pp=pp1*(1.-fract)+pp2*fract

      return
      end
