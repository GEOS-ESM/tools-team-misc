      subroutine gridcheck_nests(error)
C                                  o
********************************************************************************
*                                                                              *
*     This routine checks the grid specification for the nested model domains. *
*     It is similar to subroutine gridcheck, which checks the mother domain.   *
*                                                                              *
*     Authors: A. Stohl, G. Wotawa                                             *
*                                                                              *
*     30 December 1998                                                         *
********************************************************************************

      include 'includepar'
      include 'includecom'

      logical error

      if (numbnests.ne.0)
     +    stop 'FLEXTRA GFS cannot be operated with nested windfields'

      end
