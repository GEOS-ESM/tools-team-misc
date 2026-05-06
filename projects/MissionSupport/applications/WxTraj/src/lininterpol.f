      subroutine lininterpol(yy,nxmax,nymax,nzmax,nx,ny,nz,memind,
     +height,xt,yt,zt,itime1,itime2,itime,indexf,yint)
C                            i    i     i     i   i  i  i    i 
C       i    i  i  i    i      i      i     i    o
*****************************************************************************
*                                                                           *
*  Interpolation of 3-dimensional meteorological fields.                    *
*  In horizontal direction bilinear interpolation interpolation is used.    *
*  In the vertical and temporally a linear interpolation is used.           *
*                                                                           *
*  1,1 is the level below current position for the first time               *
*  1,2 is the level above current position for the first time               *
*  2,1 is the level below current position for the second time              *
*  2,2 is the level above current position for the second time              *
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
* indexf                indicates the number of the wind field to be read in *
* indexfh               help variable                                        *
* indz                 the level closest to the current trajectory position *
* indzh                help variable                                        *
* itime                current time                                         *
* itime1               time of the first wind field                         *
* itime2               time of the second wind field                        *
* ix,jy                x,y coordinates of lower left subgrid point          *
* memind(3)            points to the places of the wind fields              *
* nx,ny,nz             actual field dimensions in x,y and z direction       *
* nxmax,nymax,nzmax    maximum field dimensions in x,y and z direction      *
* xt                   current x coordinate                                 *
* yint                 the final interpolated value                         *
* yt                   current y coordinate                                 *
* yy(0:nxmax-1,0:nymax-1,nzmax,3)meteorological field used for interpolation*
* zt                   current z coordinate                                 *
*                                                                           *
*****************************************************************************

      implicit none

      integer nx,ny,nz,nxmax,nymax,nzmax,memind(3),i,m,n,ix,jy,ixp,jyp
      integer itime,itime1,itime2,indexf,indz,indexfh,indzh
      real yy(0:nxmax-1,0:nymax-1,nzmax,3),height(nzmax)
      real ddx,ddy,rddx,rddy,dz1,dz2,dt1,dt2,y(2),yh(2)
      real xt,yt,zt,yint
     

C Determine the level below the current position
************************************************

      do 5 i=1,nz-1
        if ((height(i).le.zt).and.(height(i+1).ge.zt)) then
          indz=i
          goto 6
        endif
5       continue
6     continue


C If point at border of grid -> small displacement into grid
************************************************************

      if (xt.ge.float(nx-1)) xt=float(nx-1)-0.00001
      if (yt.ge.float(ny-1)) yt=float(ny-1)-0.00001



***********************************************************************
C 1.) Bilinear horizontal interpolation
C This has to be done separately for 6 fields (Temporal(2)*Vertical(3))
***********************************************************************

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
     

C Vertical distance to the level below and above current position
*****************************************************************

      dz1=zt-height(indz)
      dz2=height(indz+1)-zt


C Loop over 2 time steps and 2 levels
*************************************
 
      do 10 m=1,2
        indexfh=memind(indexf+m-1)
        do 20 n=1,2
          indzh=indz+n-1

20        y(n)=rddx*rddy*yy(ix ,jy ,indzh,indexfh)
     +        + ddx*rddy*yy(ixp,jy ,indzh,indexfh)
     +        +rddx* ddy*yy(ix ,jyp,indzh,indexfh)
     +        + ddx* ddy*yy(ixp,jyp,indzh,indexfh)



***********************************
C 2.) Linear vertical interpolation
***********************************

10      yh(m)=(dz2*y(1)+dz1*y(2))/(dz1+dz2)

         

*************************************
C 3.) Temporal interpolation (linear)
*************************************

      dt1=float(itime-itime1)
      dt2=float(itime2-itime)

      yint=(yh(1)*dt2+yh(2)*dt1)/(dt1+dt2)


      return
      end
