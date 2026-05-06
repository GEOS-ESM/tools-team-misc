      subroutine levinterpol(yy,nxmax,nymax,nzmax,nx,ny,memind,
     +xt,yt,level,itime1,itime2,itime,indexf,yint)
C                               i    i     i     i   i  i    i
C     i  i    i     i      i      i     i    o
*****************************************************************************
*                                                                           *
*  Interpolation of 3-dimensional meteorological fields.                    *
*  In horizontal direction bicubic interpolation interpolation is used.     *
*  In the temporal direction linear interpolation is used.                  *
*  These interpolation techniques have been found to be most accurate.      *
*                                                                           *
*  The interpolation routines have been taken from:                         *
*  Press W.H. et al. (1992): Numerical Recipes in FORTRAN. The art of       *
*  scientific computing. 2nd edition. Cambridge University Press.           *
*                                                                           *
*  But they have been modified for faster performance.                      *
*                                                                           *
*  4-3                                                                      *
*  | |  The points are numbered in this order. Values and gradients are     *
*  1-2  stored in fields with dimension 4.                                  *
*                                                                           *
*  1 is the first time                                                      *
*  2 is the second time                                                     *
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
* indexf                indicates the number of the wind field to be read in *
* indexfh               help variable                                        *
* level                help variable                                        *
* itime                current time                                         *
* itime1               time of the first wind field                         *
* itime2               time of the second wind field                        *
* ix,jy                x,y coordinates of lower left subgrid point          *
* level                level for which interpolation shall be done          *
* memind(3)            points to the places of the wind fields              *
* nx,ny                actual field dimensions in x,y and z direction       *
* nxmax,nymax,nzmax    maximum field dimensions in x,y and z direction      *
* x1l,x2l              x,y coordinates of lower left subgrid point          *
* x1u,x2u              x,y coordinates of upper right subgrid point         *
* xt                   current x coordinate                                 *
* y(4,2,1)             subset of 4 points for 2 times and 1 level           *
* ygx(4,2,1)           x gradients at 4 points for 2 times and 1 level      *
* ygy(4,2,1)           y gradients at 4 points for 2 times and 1 level      *
* ygxy(4,2,1)          x,y gradients at 4 points for 2 times and 1 level    *
* yhelp(2,1)           the interpolated values for 2 times and 1 level      *
* yint                 the final interpolated value                         *
* yt                   current y coordinate                                 *
* yy(0:nxmax,0:nymax,nzmax,3) meteorological field used for interpolation   *
*                                                                           *
*****************************************************************************

      implicit none

      integer nx,ny,nxmax,nymax,nzmax,memind(3),i,j,l,m,n,ix,jy
      integer level,itime,itime1,itime2,indexf,indexfh,im,ip,jm,jp
      real yy(0:nxmax-1,0:nymax-1,nzmax,3)
      real y(4,2,1),ygx(4,2,1),ygy(4,2,1),ygxy(4,2,1),yhelp(2,1)
      real x1l,x1u,x2l,x2u,dt1,dt2
      real xt,yt,yint
     


C If point at border of grid -> small displacement into grid
************************************************************

      if (xt.ge.float(nx-1)) xt=float(nx-1)-0.00001
      if (yt.ge.float(ny-1)) yt=float(ny-1)-0.00001



***********************************************************************
C 1.) Bicubic horizontal interpolation
C This has to be done separately for 2 fields (Temporal)
***********************************************************************

C Determine the lower left corner
********************************* 

      ix=int(xt)
      jy=int(yt)

      x1l=float(ix)
      x1u=float(ix+1)
      x2l=float(jy)
      x2u=float(jy+1)


C Loop over the 2*2 grid points
*******************************

      do 10 l=1,4
        if (l.eq.1) then
          i=ix
          j=jy
        else if (l.eq.2) then
          i=ix+1
          j=jy
        else if (l.eq.3) then
          i=ix+1
          j=jy+1
        else if (l.eq.4) then
          i=ix
          j=jy+1
        endif
        ip=i+1
        im=i-1
        jp=j+1
        jm=j-1


C Loop over 2 time steps and 3 levels
*************************************
 
        do 10 m=1,2
        do 10 n=1,1        ! this loop and dimension is necessary to be in 
                           ! agreement with the full 3-d interpolation

          indexfh=memind(indexf+m-1)


C Values at the 2*2 subgrid
***************************

          y(l,m,n)=yy(i,j,level,indexfh)   


C Calculate derivatives in x-direction on the 2*2 subgrid
*********************************************************

          if (i.eq.0) then
            ygx(l,m,n) = yy(ip,j ,level,indexfh)
     +                 - yy(i ,j ,level,indexfh)
          else if (i.eq.nx-1) then
            ygx(l,m,n) = yy(i ,j ,level,indexfh)
     +                 - yy(im,j ,level,indexfh)
          else
            ygx(l,m,n) =(yy(ip,j ,level,indexfh)
     +                 - yy(im,j ,level,indexfh))/2.
          endif


C Calculate derivatives in y-direction on the 2*2 subgrid
*********************************************************

          if (j.eq.0) then
            ygy(l,m,n) = yy(i ,jp,level,indexfh)
     +                 - yy(i ,j ,level,indexfh)
          else if (j.eq.ny-1) then
            ygy(l,m,n) = yy(i ,j ,level,indexfh)
     +                 - yy(i ,jm,level,indexfh)
          else
            ygy(l,m,n) =(yy(i ,jp,level,indexfh)
     +                 - yy(i ,jm,level,indexfh))/2.
          endif


C Calculate cross derivative on the 2*2 subgrid
***********************************************

          if ((i.eq.0).and.(j.eq.0)) then
            ygxy(l,m,n)= yy(ip,jp,level,indexfh)-
     +                   yy(ip,j ,level,indexfh)-
     +                   yy(i ,jp,level,indexfh)+
     +                   yy(i ,j ,level,indexfh)
          else if ((i.eq.nx-1).and.(j.eq.ny-1)) then
            ygxy(l,m,n)= yy(i ,j ,level,indexfh)-
     +                   yy(i ,jm,level,indexfh)-
     +                   yy(im,j ,level,indexfh)+
     +                   yy(im,jm,level,indexfh)
          else if ((i.eq.0).and.(j.eq.ny-1)) then
            ygxy(l,m,n)= yy(ip,j ,level,indexfh)-
     +                   yy(ip,jm,level,indexfh)-
     +                   yy(i ,j ,level,indexfh)+
     +                   yy(i ,jm,level,indexfh)
          else if ((i.eq.nx-1).and.(j.eq.0)) then
            ygxy(l,m,n)= yy(i ,jp,level,indexfh)-
     +                   yy(i ,j ,level,indexfh)-
     +                   yy(im,jp,level,indexfh)+
     +                   yy(im,j ,level,indexfh)
          else if (i.eq.nx-1) then
            ygxy(l,m,n)=(yy(i ,jp,level,indexfh)-
     +                   yy(i ,jm,level,indexfh)-
     +                   yy(im,jp,level,indexfh)+
     +                   yy(im,jm,level,indexfh))/2.
          else if (i.eq.0) then
            ygxy(l,m,n)=(yy(ip,jp,level,indexfh)-
     +                   yy(ip,jm,level,indexfh)-
     +                   yy(i ,jp,level,indexfh)+
     +                   yy(i ,jm,level,indexfh))/2.
          else if (j.eq.ny-1) then
            ygxy(l,m,n)=(yy(ip,j ,level,indexfh)-
     +                   yy(ip,jm,level,indexfh)-
     +                   yy(im,j ,level,indexfh)+
     +                   yy(im,jm,level,indexfh))/2.
          else if (j.eq.0) then
            ygxy(l,m,n)=(yy(ip,jp,level,indexfh)-
     +                   yy(ip,j ,level,indexfh)-
     +                   yy(im,jp,level,indexfh)+
     +                   yy(im,j ,level,indexfh))/2.
          else
            ygxy(l,m,n)=(yy(ip,jp,level,indexfh)-
     +                   yy(ip,jm,level,indexfh)-
     +                   yy(im,jp,level,indexfh)+
     +                   yy(im,jm,level,indexfh))/4.
          endif

10        continue


C Call bicubic interpolation
****************************

      call bicubic(y,ygx,ygy,ygxy,x1l,x1u,x2l,x2u,xt,yt,yhelp,2,1)



*************************************
C 2.) Temporal interpolation (linear)
*************************************

      dt1=float(itime-itime1)
      dt2=float(itime2-itime)

      yint=(yhelp(1,1)*dt2+yhelp(2,1)*dt1)/(dt1+dt2)


      return
      end
