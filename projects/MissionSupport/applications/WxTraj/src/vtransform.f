      subroutine vtransform(vint,dydt)
***********************************************************************
*                                                                     *
*             TRAJECTORY MODEL SUBROUTINE VTRANSFORM                  *
*                                                                     *
***********************************************************************
*                                                                     *
*             AUTHOR:      G. WOTAWA                                  *
*             DATE:        1994-02-14                                 *
*             LAST UPDATE: 1996-03-21                                 *
*                          Runtime optimization                       *
*                                                                     *
***********************************************************************
*                                                                     *
* DESCRIPTION: This subroutine transforms the interpolated meridional *
* wind <vint> [m/s] to <dydt> [grid units/time unit]                  *
* ythelp      help variable computed in readgrid                      *
*                                                                     *
***********************************************************************
*                                                                     *
* INPUT:                                                              *
*                                                                     *
* vint    interpolated meridional wind component [m/s]                *
*                                                                     *
***********************************************************************
*                                                                     *
* OUTPUT:                                                             *
*                                                                     *
* dydt    total differential in y direction [grid units/time unit]    *
*                                                                     *
***********************************************************************
*
      include 'includepar'
      include 'includecom'

      real vint,dydt

      dydt=vint*ythelp

      return
      end
