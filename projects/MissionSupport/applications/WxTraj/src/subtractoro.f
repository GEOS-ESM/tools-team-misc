      subroutine subtractoro()
********************************************************************************
*                                                                              *
*   This routine subtracts the height of the orography from the height above   *
*   sea level to give height above the model ground.                           *
*                                                                              *
*     Author: A. Stohl                                                         *
*                                                                              *
*     30 April 1994                                                            *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* dist                           distance of starting point to grid points     *
* maxnests                       maximum number of nesting levels              *
* ngrid                          nesting level to be used                      *
* numpoint                       actual number of starting points              *
* oro(0:nxmax-1,0:nymax-1)       orography (mother domain)                     *
* oron(0:nxmaxn-1,0:nymaxn-1,maxnests)  orography (nested grids)               *
* orohelp                        interpolated orography                        *
* sweight                        weight for linear interpolation               *
* xpoint,ypoint,zpoint(maxpoint) x,y,z coordinates of starting points          *
*                                                                              *
* Constants:                                                                   *
* maxpoint                       maximum number of starting points             *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer j,k,ngrid
      real orohelp,xtn,ytn

      
      do 10 k=1,numpoint          ! loop over all starting points

C Only subtract for those trajectories with z coordinates given in masl
***********************************************************************

        if (kindz(k).eq.1) then

C Determine which nesting level to be used
******************************************

          ngrid=0
          do 12 j=numbnests,1,-1
            if ((xpoint(k).gt.xln(j)).and.(xpoint(k).lt.xrn(j)).and.
     +      (ypoint(k).gt.yln(j)).and.(ypoint(k).lt.yrn(j))) then
              ngrid=j
              goto 13
            endif
12          continue
13        continue


C Interpolate orography to position of starting point
*****************************************************

          if (ngrid.eq.0) then
            call orolininterpol(oro,nxmax,nymax,nx,ny,xpoint(k),
     +      ypoint(k),orohelp)
          else
            xtn=(xpoint(k)-xln(ngrid))*xresoln(ngrid)
            ytn=(ypoint(k)-yln(ngrid))*yresoln(ngrid)
            call orolininterpoln(oron,maxnests,nxmaxn,nymaxn,ngrid,
     +      nxn,nyn,xtn,ytn,orohelp)
          endif


C Subtract orography from height, but don't accept negative heights
*******************************************************************

          zpoint(k)=zpoint(k)-orohelp
          zpoint(k)=max(0.,zpoint(k))
        endif
10      continue

      return
      end
