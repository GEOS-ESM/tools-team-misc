      subroutine wtransform(wint,psint,zt,indwz,dzdt)
***********************************************************************
*                                                                     *
*             TRAJECTORY MODEL SUBROUTINE WTRANSFORM                  *
*                                                                     *
***********************************************************************
*                                                                     *
*             AUTHOR:      G. WOTAWA                                  *
*             DATE:        1994-02-14                                 *
*             LAST UPDATE: 1994-05-04                                 *
*                                                                     *
***********************************************************************
*                                                                     *
* DESCRIPTION: This subroutine transforms the interpolated vertical   *
* wind <wint> from Pa/s to eta coordinte (vertical wind velocity is   *
* given as d(eta)/dt * dp/d(eta) [Pa/s])                              *
*                                                                     *
***********************************************************************
*                                                                     *
* INPUT:                                                              *
*                                                                     *
* wint    interpolated vertical wind component [Pa/s]                 *
* psint   interpolated surface pressure        [Pa]                   *
* zt      vertical position of coordinate      [eta]                  *
* indwz   index which shows between which model layers trajectory is  *
*         situated                                                    *
*                                                                     *
***********************************************************************
*                                                                     *
* OUTPUT:                                                             *
*                                                                     *
* dzdt    total differential in z direction [eta/time unit]           *
*                                                                     *
***********************************************************************
*
      include 'includepar'
      include 'includecom'

      integer indwz
      real wint,psint,zt,dzdt,p1,p2,eta1,eta2
      real dpdeta,dpdeta1,dpdeta2,fract

***********************************************************************
*                                                                     *
*     CALCULATION OF dp/d(eta): CENTERED DIFFERENCES SCHEME USED      *
*                                                                     *
***********************************************************************


***********************************************************************
*                                                                     *
*      CALCULATE dp/d(eta) FOR LEVEL <indwz> (FIRST LEVEL)            *
*                                                                     *
***********************************************************************

      if(indwz.eq.1) then
         p1=akm(1)+bkm(1)*psint
         p2=akm(2)+bkm(2)*psint
         eta1=wheight(1)*zdirect
         eta2=wheight(2)*zdirect
      else
         p1=akm(indwz-1)+bkm(indwz-1)*psint
         p2=akm(indwz+1)+bkm(indwz+1)*psint
         eta1=wheight(indwz-1)*zdirect
         eta2=wheight(indwz+1)*zdirect
      endif
      dpdeta1=(p2-p1)/(eta2-eta1)

***********************************************************************
*                                                                     *
*      CALCULATE dp/d(eta) FOR LEVEL <indwz+1> (SECOND LEVEL)         *
*                                                                     *
***********************************************************************

      if(indwz+1.eq.nwz) then
         p1=akm(nwz-1)+bkm(nwz-1)*psint
         p2=akm(nwz)+bkm(nwz)*psint
         eta1=wheight(nwz-1)*zdirect
         eta2=wheight(nwz)*zdirect
      else
         p1=akm(indwz)+bkm(indwz)*psint
         p2=akm(indwz+2)+bkm(indwz+2)*psint
         eta1=wheight(indwz)*zdirect
         eta2=wheight(indwz+2)*zdirect
      endif
      dpdeta2=(p2-p1)/(eta2-eta1)

***********************************************************************
*                                                                     *
*      LINEAR INTERPOLATION BETWEEN FIRST AND SECOND LEVEL            *
*                                                                     *
***********************************************************************

      fract=(zt-wheight(indwz))/(wheight(indwz+1)-wheight(indwz))
      dpdeta=dpdeta1*(1.-fract)+dpdeta2*fract

***********************************************************************
*                                                                     *
*       TRANSFORMATION OF <wint> TO d(eta)/dt (ECMWF COORDINATES)     *
*                                                                     *
***********************************************************************

      dzdt=wint*zdirect/dpdeta

      return
      end
