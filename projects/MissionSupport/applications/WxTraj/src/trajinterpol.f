      subroutine trajinterpol(numb,ldimi,orotra,orotraint)          
C                              i     o
********************************************************************************
*                                                                              *
*     This routine interpolates the trajectories to a constant time step.      *
*                                                                              *
*     Author: A. Stohl                                                         *
*                                                                              *
*     11 February 1994                                                         *
*                                                                              *
*     27 February 1999 Correction by Wuyin Lin to the cyclic                   *
*     boundary conditions                                                      *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* numb                 number of trajectory to be interpolated                 *
*                                                                              *
*                                                                              *
* Constants:                                                                   *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer numb,i,j,k,ldimi
      real orotra(maxtime),orotraint(maxitime),xtmp1,xtmp2

      ldimi=1
      do 10 i=1,ldim
10      ittraint(i)=ittra(numb,1)+(i-1)*ldirect*interstep

      xtraint(1)=xtra(numb,1)
      ytraint(1)=ytra(numb,1)
      ztraint(1)=ztra(numb,1)
      ptraint(1)=ptra(numb,1)
      htraint(1)=htra(numb,1)
      qqtraint(1)=qqtra(numb,1)
      pvtraint(1)=pvtra(numb,1)
      thtraint(1)=thtra(numb,1)
      orotraint(1)=orotra(1)

      k=2
      do 20 i=2,ldim
        do 30 j=k,nttra(numb)
          if (abs(ittra(numb,j)).ge.abs(ittraint(i)))then 

C Linear interpolation
C For x coordinate, special treatment when using cyclic boundary
C conditions to keep values below 360 deg
****************************************************************

            ldimi=i
            xtmp1=xtra(numb,j-1)
            xtmp2=xtra(numb,j)
            if (xglobal) then
              if (abs(xtra(numb,j-1)-xtra(numb,j)).gt.180.) then
                if (xtra(numb,j-1).lt.xtra(numb,j)) then
                  xtmp1=xtra(numb,j-1)+360.
                else
                  xtmp2=xtra(numb,j)+360.
                endif
              endif
            endif

            xtraint(i)=(float(abs(ittra(numb,j)-ittraint(i)))*
     +      xtmp1+float(abs(ittra(numb,j-1)-ittraint(i)))*
     +      xtmp2)/float(abs(ittra(numb,j)-ittra(numb,j-1)))

            if (xtraint(i).gt.360.) xtraint(i)=xtraint(i)-360.


            ytraint(i)=(float(abs(ittra(numb,j)-ittraint(i)))*
     +      ytra(numb,j-1)+float(abs(ittra(numb,j-1)-ittraint(i)))*
     +      ytra(numb,j))/float(abs(ittra(numb,j)-ittra(numb,j-1)))

            ztraint(i)=(float(abs(ittra(numb,j)-ittraint(i)))*
     +      ztra(numb,j-1)+float(abs(ittra(numb,j-1)-ittraint(i)))*
     +      ztra(numb,j))/float(abs(ittra(numb,j)-ittra(numb,j-1)))

            ptraint(i)=(float(abs(ittra(numb,j)-ittraint(i)))*
     +      ptra(numb,j-1)+float(abs(ittra(numb,j-1)-ittraint(i)))*
     +      ptra(numb,j))/float(abs(ittra(numb,j)-ittra(numb,j-1)))

            htraint(i)=(float(abs(ittra(numb,j)-ittraint(i)))*
     +      htra(numb,j-1)+float(abs(ittra(numb,j-1)-ittraint(i)))*
     +      htra(numb,j))/float(abs(ittra(numb,j)-ittra(numb,j-1)))

            qqtraint(i)=(float(abs(ittra(numb,j)-ittraint(i)))*
     +      qqtra(numb,j-1)+float(abs(ittra(numb,j-1)-ittraint(i)))*
     +      qqtra(numb,j))/float(abs(ittra(numb,j)-ittra(numb,j-1)))

            pvtraint(i)=(float(abs(ittra(numb,j)-ittraint(i)))*
     +      pvtra(numb,j-1)+float(abs(ittra(numb,j-1)-ittraint(i)))*
     +      pvtra(numb,j))/float(abs(ittra(numb,j)-ittra(numb,j-1)))

            thtraint(i)=(float(abs(ittra(numb,j)-ittraint(i)))*
     +      thtra(numb,j-1)+float(abs(ittra(numb,j-1)-ittraint(i)))*
     +      thtra(numb,j))/float(abs(ittra(numb,j)-ittra(numb,j-1)))

            orotraint(i)=(float(abs(ittra(numb,j)-ittraint(i)))*
     +      orotra(j-1)+float(abs(ittra(numb,j-1)-ittraint(i)))*
     +      orotra(j))/float(abs(ittra(numb,j)-ittra(numb,j-1)))

            k=j

            goto 20
          endif
30        continue

20      continue

      return    
      end
