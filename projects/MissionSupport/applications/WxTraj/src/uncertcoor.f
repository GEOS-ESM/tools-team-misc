      subroutine uncertcoor()
********************************************************************************
*                                                                              *
*   This routine calculates the starting positions of the uncertainty traject. *
*                                                                              *
*     Authors: A. Stohl                                                        *
*                                                                              *
*     16 February 1994                                                         *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* compoint(maxpoint)             comment for each starting point               *
* numbunc                        number of uncertainty trajectories            *
* numpoint                       actual number of starting points              *
* phi                            angle, help variable                          *
* xpoint,ypoint,zpoint(maxpoint) x,y,z coordinates of starting points          *
*                                                                              *
* Constants:                                                                   *
* maxpoint                       maximum number of starting points             *
* pi                             PI=3.14                                       *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer i,j,k
      character*4 ii
      real phi


      do 20 i=1,numbunc
        phi=float(i)*2.*pi/float(numbunc)
        do 20 j=1,numpoint
          k=numpoint*i+j
          write(ii,'(i4.4)') i
          compoint(k)=compoint(j)(1:40)//'U'//ii
          xpoint(k)=xpoint(j)+distunc*sin(phi)
          ypoint(k)=ypoint(j)+distunc*cos(phi)
          zpoint(k)=zpoint(j)
          kind(k)=kind(j)
          kindz(k)=kindz(j)


C Check if starting positions are inside grid
*********************************************

          if (xpoint(k).gt.float(nx-1)) xpoint(k)=float(nx-1)-.00001
          if (ypoint(k).gt.float(ny-1)) ypoint(k)=float(ny-1)-.00001
          if (xpoint(k).lt.0.) xpoint(k)=.00001
          if (ypoint(k).lt.0.) ypoint(k)=.00001

20      continue

      numpoint=numpoint*(1+numbunc)

      return
      end
