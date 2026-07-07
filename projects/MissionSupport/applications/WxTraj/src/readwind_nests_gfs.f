      subroutine readwind_nests(indj,n)
C                                i   i
********************************************************************************
*                                                                              *
*     This routine reads the wind fields for the nested model domains.         *
*     It is similar to subroutine readwind, which reads the mother domain.     *
*                                                                              *
*     Authors: A. Stohl, G. Wotawa                                             *
*                                                                              *
*     30 December 1998                                                         *
********************************************************************************


      include 'includepar'
      include 'includecom'

      integer indj, n

      IF(NUMBNESTS.NE.0)
     & STOP 'SORRY: FLEXTRA (NCEP) CAN NOT BE OPERATED WITH NESTING'

      end
