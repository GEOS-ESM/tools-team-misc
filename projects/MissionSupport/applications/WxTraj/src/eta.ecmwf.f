      real function eta(psurf,pconst)
******************************************************************************
*                                                                            *
*     This routine computes vertical coordinate eta for a given pressure p.  *
*                                                                            *
*     Author: A. Stohl                                                       *
*                                                                            *
*      5 April 1994                                                          *
*      Last modification: G. Wotawa, 1994 - 04 -27                           *
*                                                                            *
******************************************************************************
* Variables:                                                                 *
* eta                   value of eta for given pressure                      *
* fract                 help variable for linear interpolation               *
* plevel1 [Pa]          pressure of level below wanted pressure              *
* plevel2 [Pa]          pressure of level above wanted pressure              *
* pconst [Pa]           pressure level of isobaric trajectory                *
* psurf [Pa]            surface level pressure                               *
* wheight(nwzmax)       height of ecmwf layer interfaces (eta coordinates)   *
******************************************************************************

      include 'includepar'
      include 'includecom'

      integer i
      real plevel1,plevel2,fract,psurf,pconst
      
      plevel1=akm(1)+bkm(1)*psurf
      if (plevel1.le.pconst) then  ! pressure higher than surface pressure
        eta=wheight(1)
      else                         ! normal case -> linear interpolation
        do i=2,nwz              ! look, between which layers we are
          plevel2=akm(i)+bkm(i)*psurf
          if (plevel2.lt.pconst) then
            fract=(pconst-plevel2)/(plevel1-plevel2)
            eta=wheight(i)*(1.-fract)+wheight(i-1)*fract
            goto 100
          endif
          plevel1=plevel2
        end do
        eta=wheight(nwz)           ! pressure lower than highest model layer
      endif

100   continue


      if (eta.gt.heightmax) eta=heightmax
      if (eta.lt.heightmin) eta=heightmin

      return
      end
