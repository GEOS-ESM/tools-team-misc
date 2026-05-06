      subroutine calcpv(n)
C
********************************************************************************
*                                                                              *
*  Calculation of potential vorticity on 3-d grid.                             *
*                                                                              *
*     Author: P. James                                                         *
*     3 February 2000                                                          *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* n                  temporal index for meteorological fields (1 to 3)         *
*                                                                              *
* Constants:                                                                   *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer n,ix,jy,i,j,k,kl,ii,jj,klvrp,klvrm,klpt,kup,kdn,kch
      integer jyvp,jyvm,ixvp,ixvm,jumpx,jumpy,jux,juy,ivrm,ivrp,ivr
      integer nlck
      real vx(2),uy(2),phi,tanphi,cosphi,dvdx,dudy,f
      real theta,thetap,thetam,dthetadp,dt1,dt2,dt,ppmk
      real height(nuvzmax),pvavr,ppml(nuvzmax)
      real thup,thdn

C Set number of levels to check for adjacent theta
      nlck=nuvz/3
      do 5 k=1,nuvz
        height(k)=akz(k)/p0+bkz(k)
5     continue
C *** Precalculate all theta levels for efficiency
        do 9 jy=0,ny-1
        do 14 kl=1,nuvz
        do 13 ix=0,nx-1
          ppmk=akz(kl)+bkz(kl)*ps(ix,jy,1,n)
          th(ix,jy,kl,n)=tt(ix,jy,kl,n)*(100000./ppmk)**kappa
13      continue
14      continue
9       continue
C
C Loop over entire grid
***********************
      do 10 jy=0,ny-1
        if (sglobal.and.jy.eq.0) goto 10
        if (nglobal.and.jy.eq.ny-1) goto 10
        phi = (ylat0 + jy * dy) * pi / 180.
        f = 0.00014585 * sin(phi)
        tanphi = tan(phi)
        cosphi = cos(phi)
C Provide a virtual jy+1 and jy-1 in case we are on domain edge (Lat)
          jyvp=jy+1
          jyvm=jy-1
          if (jy.eq.0) jyvm=0
          if (jy.eq.ny-1) jyvp=ny-1
C Define absolute gap length
          jumpy=2
          if (jy.eq.0.or.jy.eq.ny-1) jumpy=1
          if (sglobal.and.jy.eq.1) then
             jyvm=1
             jumpy=1
          end if
          if (nglobal.and.jy.eq.ny-2) then
             jyvp=ny-2
             jumpy=1
          end if
          juy=jumpy
C
        do 11 ix=0,nx-1
C Provide a virtual ix+1 and ix-1 in case we are on domain edge (Long)
          ixvp=ix+1
          ixvm=ix-1
          jumpx=2
          if (xglobal) then
             ivrp=ixvp
             ivrm=ixvm
             if (ixvm.lt.0) ivrm=ixvm+nx-1
             if (ixvp.ge.nx) ivrp=ixvp-nx+1
          else
            if (ix.eq.0) ixvm=0
            if (ix.eq.nx-1) ixvp=nx-1
            ivrp=ixvp
            ivrm=ixvm
C Define absolute gap length
            if (ix.eq.0.or.ix.eq.nx-1) jumpx=1
          end if
          jux=jumpx
C Precalculate pressure values for efficiency
          do 8 kl=1,nuvz
            ppml(kl)=akz(kl)+bkz(kl)*ps(ix,jy,1,n)
8         continue
C
C Loop over the vertical
************************

          do 12 kl=1,nuvz
            theta=th(ix,jy,kl,n)
            klvrp=kl+1
            klvrm=kl-1
            klpt=kl
C If top or bottom level, dthetadp is evaluated between the current
C level and the level inside, otherwise between level+1 and level-1
C
            if (klvrp.gt.nuvz) klvrp=nuvz
            if (klvrm.lt.1) klvrm=1
            thetap=th(ix,jy,klvrp,n)
            thetam=th(ix,jy,klvrm,n)
            dthetadp=(thetap-thetam)/(ppml(klvrp)-ppml(klvrm))
            
C Compute vertical position at pot. temperature surface on subgrid
C and the wind at that position
******************************************************************
C a) in x direction
            ii=0
            do 20 i=ixvm,ixvp,jumpx
              ivr=i
              if (xglobal) then
                 if (i.lt.0) ivr=ivr+nx-1
                 if (i.ge.nx) ivr=ivr-nx+1
              end if
              ii=ii+1
C Search adjacent levels for current theta value
C Spiral out from current level for efficiency
              kup=klpt-1
              kdn=klpt
              kch=0
40            continue
C Upward branch
              kup=kup+1
              if (kch.ge.nlck) goto 21     ! No more levels to check, 
C                                            ! and no values found
              if (kup.ge.nuvz) goto 41
              kch=kch+1
              k=kup
              thdn=th(ivr,jy,k,n)
              thup=th(ivr,jy,k+1,n)
          if (((thdn.ge.theta).and.(thup.le.theta)).or.
     +    ((thdn.le.theta).and.(thup.ge.theta))) then
                  dt1=abs(theta-thdn)
                  dt2=abs(theta-thup)
                  dt=dt1+dt2
                  if (dt.lt.eps) then   ! Avoid division by zero error
                    dt1=0.5             ! G.W., 10.4.1996
                    dt2=0.5
                    dt=1.0
                  endif
              vx(ii)=(vv(ivr,jy,k,n)*dt2+vv(ivr,jy,k+1,n)*dt1)/dt
                  goto 20
                endif
41            continue
C Downward branch
              kdn=kdn-1
              if (kdn.lt.1) goto 40
              kch=kch+1
              k=kdn
              thdn=th(ivr,jy,k,n)
              thup=th(ivr,jy,k+1,n)
          if (((thdn.ge.theta).and.(thup.le.theta)).or.
     +    ((thdn.le.theta).and.(thup.ge.theta))) then
                  dt1=abs(theta-thdn)
                  dt2=abs(theta-thup)
                  dt=dt1+dt2
                  if (dt.lt.eps) then   ! Avoid division by zero error
                    dt1=0.5             ! G.W., 10.4.1996
                    dt2=0.5
                    dt=1.0
                  endif
              vx(ii)=(vv(ivr,jy,k,n)*dt2+vv(ivr,jy,k+1,n)*dt1)/dt
                  goto 20
                endif
                goto 40
C This section used when no values were found
21          continue
C Must use vv at current level and long. jux becomes smaller by 1
            vx(ii)=vv(ix,jy,kl,n)
            jux=jux-1
C Otherwise OK
20          continue
          if (jux.gt.0) then
          dvdx=(vx(2)-vx(1))/float(jux)/(dx*pi/180.)
          else
          dvdx=vv(ivrp,jy,kl,n)-vv(ivrm,jy,kl,n)
          dvdx=dvdx/float(jumpx)/(dx*pi/180.)
C Only happens if no equivalent theta value
C can be found on either side, hence must use values
C from either side, same pressure level.
          end if

C b) in y direction

            jj=0
            do 50 j=jyvm,jyvp,jumpy
              jj=jj+1
C Search adjacent levels for current theta value
C Spiral out from current level for efficiency
              kup=klpt-1
              kdn=klpt
              kch=0
70            continue
C Upward branch
              kup=kup+1
              if (kch.ge.nlck) goto 51     ! No more levels to check, 
C                                          ! and no values found
              if (kup.ge.nuvz) goto 71
              kch=kch+1
              k=kup
              thdn=th(ix,j,k,n)
              thup=th(ix,j,k+1,n)
          if (((thdn.ge.theta).and.(thup.le.theta)).or.
     +    ((thdn.le.theta).and.(thup.ge.theta))) then
                  dt1=abs(theta-thdn)
                  dt2=abs(theta-thup)
                  dt=dt1+dt2
                  if (dt.lt.eps) then   ! Avoid division by zero error
                    dt1=0.5             ! G.W., 10.4.1996
                    dt2=0.5
                    dt=1.0
                  endif
                  uy(jj)=(uu(ix,j,k,n)*dt2+uu(ix,j,k+1,n)*dt1)/dt
                  goto 50
                endif
71            continue
C Downward branch
              kdn=kdn-1
              if (kdn.lt.1) goto 70
              kch=kch+1
              k=kdn
              thdn=th(ix,j,k,n)
              thup=th(ix,j,k+1,n)
          if (((thdn.ge.theta).and.(thup.le.theta)).or.
     +    ((thdn.le.theta).and.(thup.ge.theta))) then
                  dt1=abs(theta-thdn)
                  dt2=abs(theta-thup)
                  dt=dt1+dt2
                  if (dt.lt.eps) then   ! Avoid division by zero error
                    dt1=0.5             ! G.W., 10.4.1996
                    dt2=0.5
                    dt=1.0
                  endif
                  uy(jj)=(uu(ix,j,k,n)*dt2+uu(ix,j,k+1,n)*dt1)/dt
                  goto 50
                endif
                goto 70
C This section used when no values were found
51          continue
C Must use uu at current level and lat. juy becomes smaller by 1
            uy(jj)=uu(ix,jy,kl,n)
            juy=juy-1
C Otherwise OK
50          continue
          if (juy.gt.0) then
          dudy=(uy(2)-uy(1))/float(juy)/(dy*pi/180.)
          else
          dudy=uu(ix,jyvp,kl,n)-uu(ix,jyvm,kl,n)
          dudy=dudy/float(jumpy)/(dy*pi/180.)
          end if
C
          pv(ix,jy,kl,n)=dthetadp*(f+(dvdx/cosphi-dudy
     +    +uu(ix,jy,kl,n)*tanphi)/r_earth)*(-1.e6)*9.81
C
C Resest jux and juy
          jux=jumpx
          juy=jumpy
12        continue
11        continue                                                            
10        continue
C
C Fill in missing PV values on poles, if present
C Use mean PV of surrounding latitude ring
C
          if (sglobal) then
             do 80 kl=1,nuvz
                pvavr=0.
                do 81 ix=0,nx-1
                   pvavr=pvavr+pv(ix,1,kl,n)
81              continue
                pvavr=pvavr/float(nx)
                jy=0
                do 82 ix=0,nx-1
                   pv(ix,jy,kl,n)=pvavr
82              continue
80           continue
          end if
          if (nglobal) then
             do 90 kl=1,nuvz
                pvavr=0.
                do 91 ix=0,nx-1
                   pvavr=pvavr+pv(ix,ny-2,kl,n)
91              continue
                pvavr=pvavr/float(nx)
                jy=ny-1
                do 92 ix=0,nx-1
                   pv(ix,jy,kl,n)=pvavr
92              continue
90           continue
          end if
      return
      end
