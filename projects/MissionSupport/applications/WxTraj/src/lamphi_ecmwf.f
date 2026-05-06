      subroutine lamphi_ecmwf(xmod,ymod,xlon,ylat) 
C                              i     i   o    o
********************************************************************************
*                                                                              *
*     This routine transforms model coordinates to geografical coordinates.    *
*                                                                              *
*     Authors: A. Stohl                                                        *
*                                                                              *
*     7 April 1994                                                             *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* xmod,ymod          model coordinates                                         *
* xlon,ylat          geografical coordinates                                   *

* Constants:                                                                   *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      real xlon,ylat,xmod,ymod


      xlon=xmod*dx+xlon0
      ylat=ymod*dy+ylat0

      return
      end
