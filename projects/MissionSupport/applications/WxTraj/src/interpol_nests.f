      subroutine interpoln(yy,maxnests,nxmax,nymax,nzmax,ngrid,
     +nxn,nyn,nz,memind,height,xt,yt,zt,itime1,itime2,itime,indexf,yint)
C                          i     i       i     i     i     i
C      i   i  i    i      i    i  i  i    i      i      i     i     i
*****************************************************************************
*                                                                           *
*  Interpolation of nested 3-dimensional meteorological fields.             *
*  In horizontal direction bicubic interpolation interpolation is used.     *
*  In the vertical a polynomial interpolation is used.                      *
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
*  1,2 is the level closest to the current position for the first time      *
*  1,1 is the level below 1,2                                               *
*  1,3 is the level above 1,2                                               *
*  2,2 is the level closest to the current position for the second time     *
*  2,1 is the level below 2,2                                               *
*  2,3 is the level above 2,2                                               *
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
* indexf               indicates the number of the wind field to be read in *
* indexfh              help variable                                        *
* indz                 the level closest to the current trajectory position *
* indzh                help variable                                        *
* itime                current time                                         *
* itime1               time of the first wind field                         *
* itime2               time of the second wind field                        *
* ix,jy                x,y coordinates of lower left subgrid point          *
* memind(3)            points to the places of the wind fields              *
* nx,ny,nz             actual field dimensions in x,y and z direction       *
* nxmax,nymax,nzmax    maximum field dimensions in x,y and z direction      *
* x1l,x2l              x,y coordinates of lower left subgrid point          *
* x1u,x2u              x,y coordinates of upper right subgrid point         *
* xt                   current x coordinate                                 *
* y(4,2,3)             subset of 4 points for 2 times and 3 levels          *
* ygx(4,2,3)           x gradients at 4 points for 2 times and 3 levels     *
* ygy(4,2,3)           y gradients at 4 points for 2 times and 3 levels     *
* ygxy(4,2,3)          x,y gradients at 4 points for 2 times and 3 levels   *
* yhelp(2,3)           the interpolated values for 2 times and 3 levels     *
* yh2(2)               the interpolated values for the 2 times              *
* yint                 the final interpolated value                         *
* yt                   current y coordinate                                 *
* yy(0:nxmax,0:nymax,nzmax,3) meteorological field used for interpolation   *
* zt                   current z coordinate                                 *
*                                                                           *
*****************************************************************************

      implicit none

      integer maxnests,nxn(maxnests),nyn(maxnests),nz,nxmax,nymax,nzmax
      integer ngrid,memind(3),i,j,l,m,n,ix,jy
      integer itime,itime1,itime2,indexf,indz,indexfh,indzh,im,ip,jm,jp
      real yy(0:nxmax-1,0:nymax-1,nzmax,3,maxnests),height(nzmax)
      real y(4,2,3),ygx(4,2,3),ygy(4,2,3),ygxy(4,2,3),yhelp(2,3),yh2(2)
      real x1l,x1u,x2l,x2u,dz1,dz2,dt1,dt2,zz(3)
      real xt,yt,zt,yint
     

C 3 levels are needed for the polynomial interpolation in the vertical
C Determine the closest vertical level -1
**********************************************************************

      do 5 i=1,nz-1
        if ((height(i).le.zt).and.(height(i+1).ge.zt)) then
          dz1=zt-height(i)
          dz2=height(i+1)-zt
          if (dz1.lt.dz2) then
            indz=i-1
          else
            indz=i
          endif
          goto 6
        endif
5       continue
6     continue
      if (indz.lt.1) indz=1
      if (indz.gt.(nz-2)) indz=nz-2


C If point at border of grid -> small displacement into grid
************************************************************

      if (xt.ge.float(nxn(ngrid)-1)) xt=float(nxn(ngrid)-1)-0.00001
      if (yt.ge.float(nyn(ngrid)-1)) yt=float(nyn(ngrid)-1)-0.00001



***********************************************************************
C 1.) Bicubic horizontal interpolation
C This has to be done separately for 6 fields (Temporal(2)*Vertical(3))
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
          indexfh=memind(indexf+m-1)
          do 10 n=1,3
          indzh=indz+n-1


C Values at the 2*2 subgrid
***************************

            y(l,m,n)=yy(i,j,indzh,indexfh,ngrid)   


C Calculate derivatives in x-direction on the 2*2 subgrid
*********************************************************

            if (i.eq.0) then
              ygx(l,m,n) = yy(ip,j ,indzh,indexfh,ngrid)
     +                   - yy(i ,j ,indzh,indexfh,ngrid)
            else if (i.eq.nxn(ngrid)-1) then
              ygx(l,m,n) = yy(i ,j ,indzh,indexfh,ngrid)
     +                   - yy(im,j ,indzh,indexfh,ngrid)
            else
              ygx(l,m,n) =(yy(ip,j ,indzh,indexfh,ngrid)
     +                   - yy(im,j ,indzh,indexfh,ngrid))/2.
            endif


C Calculate derivatives in y-direction on the 2*2 subgrid
*********************************************************

            if (j.eq.0) then
              ygy(l,m,n) = yy(i ,jp,indzh,indexfh,ngrid)
     +                   - yy(i ,j ,indzh,indexfh,ngrid)
            else if (j.eq.nyn(ngrid)-1) then
              ygy(l,m,n) = yy(i ,j ,indzh,indexfh,ngrid)
     +                   - yy(i ,jm,indzh,indexfh,ngrid)
            else
              ygy(l,m,n) =(yy(i ,jp,indzh,indexfh,ngrid)
     +                   - yy(i ,jm,indzh,indexfh,ngrid))/2.
            endif


C Calculate cross derivative on the 2*2 subgrid
***********************************************

            if ((i.eq.0).and.(j.eq.0)) then
              ygxy(l,m,n)= yy(ip,jp,indzh,indexfh,ngrid)-
     +                     yy(ip,j ,indzh,indexfh,ngrid)-
     +                     yy(i ,jp,indzh,indexfh,ngrid)+
     +                     yy(i ,j ,indzh,indexfh,ngrid)
            else if ((i.eq.nxn(ngrid)-1).and.(j.eq.nyn(ngrid)-1)) then
              ygxy(l,m,n)= yy(i ,j ,indzh,indexfh,ngrid)-
     +                     yy(i ,jm,indzh,indexfh,ngrid)-
     +                     yy(im,j ,indzh,indexfh,ngrid)+
     +                     yy(im,jm,indzh,indexfh,ngrid)
            else if ((i.eq.0).and.(j.eq.nyn(ngrid)-1)) then
              ygxy(l,m,n)= yy(ip,j ,indzh,indexfh,ngrid)-
     +                     yy(ip,jm,indzh,indexfh,ngrid)-
     +                     yy(i ,j ,indzh,indexfh,ngrid)+
     +                     yy(i ,jm,indzh,indexfh,ngrid)
            else if ((i.eq.nxn(ngrid)-1).and.(j.eq.0)) then
              ygxy(l,m,n)= yy(i ,jp,indzh,indexfh,ngrid)-
     +                     yy(i ,j ,indzh,indexfh,ngrid)-
     +                     yy(im,jp,indzh,indexfh,ngrid)+
     +                     yy(im,j ,indzh,indexfh,ngrid)
            else if (i.eq.nxn(ngrid)-1) then
              ygxy(l,m,n)=(yy(i ,jp,indzh,indexfh,ngrid)-
     +                     yy(i ,jm,indzh,indexfh,ngrid)-
     +                     yy(im,jp,indzh,indexfh,ngrid)+
     +                     yy(im,jm,indzh,indexfh,ngrid))/2.
            else if (i.eq.0) then
              ygxy(l,m,n)=(yy(ip,jp,indzh,indexfh,ngrid)-
     +                     yy(ip,jm,indzh,indexfh,ngrid)-
     +                     yy(i ,jp,indzh,indexfh,ngrid)+
     +                     yy(i ,jm,indzh,indexfh,ngrid))/2.
            else if (j.eq.nyn(ngrid)-1) then
              ygxy(l,m,n)=(yy(ip,j ,indzh,indexfh,ngrid)-
     +                     yy(ip,jm,indzh,indexfh,ngrid)-
     +                     yy(im,j ,indzh,indexfh,ngrid)+
     +                     yy(im,jm,indzh,indexfh,ngrid))/2.
            else if (j.eq.0) then
              ygxy(l,m,n)=(yy(ip,jp,indzh,indexfh,ngrid)-
     +                     yy(ip,j ,indzh,indexfh,ngrid)-
     +                     yy(im,jp,indzh,indexfh,ngrid)+
     +                     yy(im,j ,indzh,indexfh,ngrid))/2.
            else
              ygxy(l,m,n)=(yy(ip,jp,indzh,indexfh,ngrid)-
     +                     yy(ip,jm,indzh,indexfh,ngrid)-
     +                     yy(im,jp,indzh,indexfh,ngrid)+
     +                     yy(im,jm,indzh,indexfh,ngrid))/4.
            endif

10          continue


C Call bicubic interpolation
****************************

      call bicubic(y,ygx,ygy,ygxy,x1l,x1u,x2l,x2u,xt,yt,yhelp,2,3)



**************************************************
C 2. Vertical interpolation by a 2nd order polynom
**************************************************

      do 20 n=1,3
20      zz(n)=height(indz+n-1)

      call polynom(zz,yhelp,3,zt,yh2,2)



*************************************
C 3.) Temporal interpolation (linear)
*************************************

      dt1=float(itime-itime1)
      dt2=float(itime2-itime)

      yint=(yh2(1)*dt2+yh2(2)*dt1)/(dt1+dt2)


      return
      end
