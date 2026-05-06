      subroutine geteta(xt,yt,eta,theta,itime1,itime2,itime,indexf,
     +psurf,ngrid,indz1,indz2)        
C                       i  i  i/o   i     i      i      i     i
C       i     i     o     o
*****************************************************************************
*                                                                           *
*     Calculates eta for a given potential temperature.                     *
*     If profile is unstable, an eta region is given instead of a single    *
*     level.                                                                *
*                                                                           *
*     Author: A. Stohl                                                      *
*                                                                           *
*     27 April 1994                                                         *
*                                                                           *
*     Modified by Petra Seibert, 27 April 1995:                             *
*     indz1 and indz2 will contain indz on output instead of 1              *
*     in stable  case                                                       *
*                                                                           *
*****************************************************************************
*                                                                           *
* Variables:                                                                *
* akz,bkz(nuvzmax)     coefficients for computing the heights of the eta lev*
* deltazt              vertical displacement of trajectory in eta coordinate*
* dt,dt1,dt2           weighting factors                                    *
* eta                  old eta level of isentropic trajectory               *
* indexf               indicates the number of the wind field to be read in *
* indz1,indz2          indices of layer boundaries of unstable region       *
* itime                time index of trajectory position                    *
* itime1               time index of first temperature field                *
* itime2               time index of second temperature field               *
* ngrid                level of nested grid to be used                      *
* nx,ny,nuvz           field dimensions in x,y and z for (u,v) direction    *
* pp(nuvzmax) [Pa]     Pressure at the model levels                         *
* psurf [Pa]           Surface pressure                                     *
* theta [K]            theta level of trajectory calculation                *
* tt(0:nxmax-1,0:nymax-1,nuvzmax,3)  3-dim temperature field                *
* ttlev(nuvzmax) [K]   Potential temperature at the model levels            *
* xt,yt                Position of trajectory                               *
*                                                                           *
* Constants:                                                                *
* kappa                exponent for calculating potential temperature       *
* nuvzmax              maximum number of levels, where u,v and tt are given *
*                                                                           *
*****************************************************************************

      include 'includepar'
      include 'includecom'

      integer itime,itime1,itime2,indexf,k,indz,indz1,indz2,i,ngrid
      real xt,yt,zt,theta,zthelp,dt,dt1,dt2,deltazt,eta,thelp
      real psurf,pp(nuvzmax),ttlev(nuvzmax)


*********************************************
C 1. Calculate the pressure at all eta levels
********************************************* 

      do k=1,nuvz
        pp(k)=akz(k)+bkz(k)*psurf
      end do

************************************************
C 2. Calculate the temperature at all eta levels
************************************************

      do 16 k=1,nuvz
       if (ngrid.gt.0) then
         call levlininterpoln(ttn,maxnests,nxmaxn,nymaxn,nuvzmax,ngrid,
     +   nxn,nyn,memind,xt,yt,k,itime1,itime2,itime,indexf,ttlev(k))
       else
         call levlininterpol(tt,nxmax,nymax,nuvzmax,nx,ny,memind,
     +   xt,yt,k,itime1,itime2,itime,indexf,ttlev(k))
       endif
16     continue


**********************************************************
C 3. Calculate the potential temperature at all eta levels
**********************************************************

      do k=1,nuvz
        ttlev(k)=ttlev(k)*(100000/pp(k))**kappa
      end do

*****************************************************************************
C 4. Search for the potential temperature that is equal to the temperature of
C trajectory calculation. If the region where this temperature can be found
C is unstable, find the lower and upper boundary of the unstable region.
*****************************************************************************

C Find the height where potential temp. is equal to trajectory level.
C If more levels exist, where this is valid, search for the one
C closest in eta coordinate to the eta coordinate of last time step.
*********************************************************************

      deltazt=999999.
      do 30 k=1,nuvz-1
        if (((ttlev(k).ge.theta).and.(ttlev(k+1).le.theta)).or.
     +  ((ttlev(k).le.theta).and.(ttlev(k+1).ge.theta))) then
          dt1=abs(theta-ttlev(k))
          dt2=abs(theta-ttlev(k+1))
          dt=dt1+dt2
          if (dt.lt.eps) then   ! Avoid division by zero error
            dt1=0.5             ! G.W., 10.4.1996
            dt2=0.5
            dt=1.0
          endif
          zthelp=(uvheight(k)*dt2+uvheight(k+1)*dt1)/dt
          if (abs(zthelp-eta).lt.deltazt) then
            deltazt=abs(zthelp-eta)
            zt=zthelp
            indz=k
          endif
        endif
30      continue


      if (deltazt.gt.999990.) then                 ! no level has been found
        if (abs(ttlev(1)-theta).lt.abs(ttlev(nuvz)-theta)) then
          indz=1
          zt=uvheight(1)
        else
          indz=nuvz-1
          zt=uvheight(nuvz)
        endif
      endif


C Determine if the air at trajectory level is stable or unstable stratified
C If it is stable -> just take the level
C If it is unstable -> take the whole unstable region and determine lower
C and upper boundary of this region
***************************************************************************

      if (ttlev(indz).lt.ttlev(indz+1)) then     ! stable region
        eta=zt
        indz1=indz       
        indz2=indz1       ! -----, indz1 and indz2 must be equal
      else                                       ! unstable region
        indz1=1
        do 40 k=indz-1,1,-1
          if (ttlev(k).lt.ttlev(k+1)) then
            indz1=k+1                     ! lower boundary of unstable region
            thelp=ttlev(indz1)
            goto 45
          endif
40        continue
45      continue
        indz2=indz+1
        do 50 k=indz+2,nuvz
          if (ttlev(k).gt.ttlev(indz1)) then
            indz2=k-1                     ! upper boundary of unstable region
            goto 55
          endif
50        continue
55      continue
        eta=uvheight(indz2)


C Calculate new mean potential temperature of the unstable layer
****************************************************************

        theta=0.
        do i=indz1,indz2
          theta=theta+ttlev(i)
        end do
        theta=theta/float(indz2-indz1+1)
      endif


      return
      end
