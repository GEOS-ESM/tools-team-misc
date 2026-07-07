      subroutine levlininterpoln(yy,maxnests,nxmax,nymax,nzmax,ngrid,
     +nxn,nyn,memind,xt,yt,level,itime1,itime2,itime,index,yint)
C                                i     i       i     i     i     i
C      i   i    i    i  i    i     i      i      i     i    i
*****************************************************************************
*                                                                           *
*  Interpolation of meteorological fields on 2-d model layers.              *
*  In horizontal direction bilinear interpolation interpolation is used.    *
*  Temporally a linear interpolation is used.                               *
*                                                                           *
*  1 first time                                                             *
*  2 second time                                                            *
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
* dt1,dt2              time differences between fields and current position *
* dz1,dz2              z distance between levels and current position       *
* height(nzmax)        heights of the model levels                          *
* index                indicates the number of the wind field to be read in *
* indexh               help variable                                        *
* indz                 the level closest to the current trajectory position *
* indzh                help variable                                        *
* itime                current time                                         *
* itime1               time of the first wind field                         *
* itime2               time of the second wind field                        *
* ix,jy                x,y coordinates of lower left subgrid point          *
* level                level at which interpolation shall be done           *
* memind(3)            points to the places of the wind fields              *
* nx,ny                actual field dimensions in x,y and z direction       *
* nxmax,nymax,nzmax    maximum field dimensions in x,y and z direction      *
* xt                   current x coordinate                                 *
* yint                 the final interpolated value                         *
* yt                   current y coordinate                                 *
* yy(0:nxmax,0:nymax,nzmax,3) meteorological field used for interpolation   *
* zt                   current z coordinate                                 *
*                                                                           *
*****************************************************************************

      implicit none

      integer maxnests,nxn(maxnests),nyn(maxnests),nxmax,nymax,nzmax
      integer ngrid,memind(3),m,ix,jy,ixp,jyp
      integer itime,itime1,itime2,level,index,indexh
      real yy(0:nxmax-1,0:nymax-1,nzmax,3,maxnests)
      real ddx,ddy,rddx,rddy,dt1,dt2,y(2)
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
 
      do 10 m=1,2
        indexh=memind(index+m-1)

10      y(m)=rddx*rddy*yy(ix ,jy ,level,indexh,ngrid)
     +      + ddx*rddy*yy(ixp,jy ,level,indexh,ngrid)
     +      +rddx* ddy*yy(ix ,jyp,level,indexh,ngrid)
     +      + ddx* ddy*yy(ixp,jyp,level,indexh,ngrid)


*************************************
C 2.) Temporal interpolation (linear)
*************************************

      dt1=float(itime-itime1)
      dt2=float(itime2-itime)

      yint=(y(1)*dt2+y(2)*dt1)/(dt1+dt2)


      return
      end
