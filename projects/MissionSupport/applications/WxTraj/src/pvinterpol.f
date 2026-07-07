      subroutine pvinterpol 
     &    (pv, xt,yt,itime1,itime2,itime, pvint)
C           i   i  i     i     i    i       o

C  Special interpolation subroutine for pot. vorticity on a small subgrid   
C  In horizontal direction bilinear interpolation interpolation is used.    
C  Temporally a linear interpolation is used.                               
C                                                                           
C  Variables:
C  pv (nx,ny,nt)   potential vorticity on 2*2*2 subgrid
C  xt,yt           horizontal position in interval [0:1,0:1]
C                  to which the interpolation is to be made
C  itime1,itime2   time in seconds for which PV is given
C  itime           time in seconds for which PV is desired
C  pvint           interpolated value of PV
C
C     Author: Petra Seibert                                                 
C     based on subroutine lininterpol.f by A. Stohl from  30 May 1994     
C     Date: 3 May 1995

      dimension pv(2,2,2), pvt(2) 

C     bilinear horizontal interpolation

      ddx = xt
      ddy = yt

      rddx = 1.-ddx
      rddy = 1.-ddy

C     loop over 2 time levels

      do 20 n=1,2
        pvt(n) = rddx * rddy * pv(1,1,n)
     +          + ddx * rddy * pv(2,1,n)
     +          +rddx * ddy  * pv(1,2,n)
     +          + ddx * ddy  * pv(2,2,n)
20      continue

C     temporal interpolation (linear)

      dt1 = float( itime - itime1 )
      dt2 = float( itime2 - itime )
      pvint = (pvt(1) * dt2 + pvt(2) * dt1) / (dt1 + dt2)

      return
      end
