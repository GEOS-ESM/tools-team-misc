      subroutine orolininterpoln(yy,maxnests,nxmax,nymax,ngrid,nxn,nyn,
     +xt,yt,yint)
C                                i     i       i     i     i    i   i
C     i  i   o
*****************************************************************************
*                                                                           *
*  Interpolation of nested meteorological fields on 2-d model layers.       *
*  A bilinear interpolation interpolation is used.                          *
*                                                                           *
*                                                                           *
*     Author: A. Stohl                                                      *
*                                                                           *
*     30 May 1994                                                           *
*                                                                           *
*****************************************************************************
*                                                                           *
* Variables:                                                                *
*                                                                           *
* height(nzmax)        heights of the model levels                          *
* ix,jy                x,y coordinates of lower left subgrid point          *
* maxnests             maximum allowed number of nests                      *
* ngrid                currently used number of nests                       *
* nxn,nyn              actual field dimensions in x,y and z direction       *
* nxmax,nymax,nzmax    maximum field dimensions in x,y and z direction      *
* xt                   current x coordinate                                 *
* yint                 the final interpolated value                         *
* yt                   current y coordinate                                 *
* yy(0:nxmax-1,0:nymax-1) meteorological field used for interpolation       *
* zt                   current z coordinate                                 *
*                                                                           *
*****************************************************************************

      implicit none

      integer maxnests,nxn(maxnests),nyn(maxnests),nxmax,nymax,ngrid
      integer ix,jy,ixp,jyp
      real yy(0:nxmax-1,0:nymax-1,maxnests)
      real ddx,ddy,rddx,rddy
      real xt,yt,yint
     


C If point at border of grid -> small displacement into grid
************************************************************

      if (xt.ge.float(nxn(ngrid)-1)) xt=float(nxn(ngrid)-1)-0.00001
      if (yt.ge.float(nyn(ngrid)-1)) yt=float(nyn(ngrid)-1)-0.00001



***********************************************************************
C 1.) Bilinear horizontal interpolation
C This has to be done separately for 2 fields (Temporal)
********************************************************

C Determine the lower left corner and its distance to the current position
************************************************************************** 

      ix=int(xt)
      jy=int(yt)
      ixp=ix+1
      jyp=jy+1
      ddx=xt-float(ix)
      ddy=yt-float(jy)
      rddx=1.-ddx
      rddy=1.-ddy
     

C Loop over 2 time steps
************************
 
      yint=rddx*rddy*yy(ix ,jy ,ngrid)
     +    + ddx*rddy*yy(ixp,jy ,ngrid)
     +    +rddx* ddy*yy(ix ,jyp,ngrid)
     +    + ddx* ddy*yy(ixp,jyp,ngrid)



      return
      end
