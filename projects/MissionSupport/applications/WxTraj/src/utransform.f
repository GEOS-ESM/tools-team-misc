      subroutine utransform(uint,yt,dxdt)
***********************************************************************
*                                                                     *
*             TRAJECTORY MODEL SUBROUTINE UTRANSFORM                  *
*                                                                     *
***********************************************************************
*                                                                     *
*             AUTHOR:      G. WOTAWA                                  *
*             DATE:        1994-02-14                                 *
*             LAST UPDATE: 1996-03-21 A. Stohl                        *
*                          Runtime optimization                       *
*                                                                     *
*                                                                     *
***********************************************************************
*                                                                     *
* DESCRIPTION: This subroutine transforms the interpolated zonal      *
* wind <uint> [m/s] to <dxdt> [grid units/time unit]                  *
* xthelp  help variable computed in readgrid                          *
*                                                                     *
***********************************************************************
*                                                                     *
* INPUT:                                                              *
*                                                                     *
* uint    interpolated zonal wind component [m/s]                     *
*                                                                     *
***********************************************************************
*                                                                     *
* OUTPUT:                                                             *
*                                                                     *
* dxdt    total differential in x direction [grid units/time unit]    *
*                                                                     *
***********************************************************************
*
      include 'includepar'
      include 'includecom'

      real uint,yt,dxdt,fact,pih
      parameter(pih=pi/180.)

      fact=max(cos((yt*dy+ylat0)*pih),1.e-4)

      dxdt=uint/fact*xthelp

      return
      end
