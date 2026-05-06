      subroutine petters(minstep,nt,itime,xold,yold,zold,pold,hold,
     +qqold,pvold,thold,xnew,ynew,znew,pnew,hnew,qqnew,pvnew,thnew,
     +ntstep,nstop,levconst,init,lkind,lkindz,unc,trerroru,
     +trerrorv,trerrorw)
C                           i    i    i    i    i    i    o    o
C       o      o    o    o    o    o    o    o     o     o     o
C       o      o      i      i     i     i     i     o
C        o        o
********************************************************************************
*                                                                              *
*  Calculation of trajectories utilizing the Petterssen scheme.                *
*  The time step is determined by the Courant-Friedrichs-Lewy (CFL) criterion. *
*  This means that the time step must be so small that the displacement within *
*  this time step is smaller than 1 grid distance. Additionally, a temporal    *
*  CFL criterion is introduced: the time step must be smaller than the time    *
*  interval of the wind fields used for interpolation.                         *
*  The CFL criterion is only used for the initialization of the iteration      *
*  scheme. Within the Petterssen iteration scheme this time step is not        *
*  corrected, but kept constant.                                               *
*                                                                              *
*     Author: A. Stohl                                                         *
*                                                                              *
*     1 February 1994                                                          *
*                                                                              *
*     Update: 16 February 1997: limitation of the estimated truncation error   *
*                                                                              *
*     Update: 29 July 1998: use of global data                                 *
*                                                                              *
*  Literature:                                                                 *
*  Petterssen (1940): Weather Analysis and Forecasting. McGraw-Hill Book       *
*                     Company, Inc.                                            *
*  Seibert P. (1993): Convergence and Accuracy of Numerical Methods for        *
*                     Trajectory Calculation. J. Appl. Met. 32, 558-566.       *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* corr               correlation coefficient of random errors                  *
* epsu,epsv,epsw     magnitude of random errors for uncertainty trajecories    *
* idiff [s]          Temporal distance between the windfields used for interpol*
* init               .true. for first time step of trajectory                  *
* indwz              index of the model layer beneath current position of traj.*
* itermax            maximum number of iterations used in the integration schem*
* itime [s]          current temporal position                                 *
* ldirect            Temporal direction of trajectories (-1=backward,1=forward)*
* levconst           height of trajectory in Pa, m or K depending on type of t.*
* lkind              kind of trajectory (e.g. isobaric, 3-dimensional)         *
* lkindz             unit of z coordinate (1:masl, 2:magl, 3:hPa)              *
* minstep [s]        minimum possible time step=not to exceed dimension maxtime*
* ngrid              index which grid is to be used                            *
* nstop              =greater 0, if trajectory calculation is finished         *
* nt [s]             Current age of trajectory                                 *
* ntstep             final time step of trajectory calculation                 *
* ntstep1            time step determined by horizontal CFL-criterion          *
* ntstep2            time step determined by vertical CFL-criterion            *
* ntstep3            time step determined by temporal CFL-criterion            *
* pnew,hnew          height at next time steps position in pressure and z coor *
* pold,hold          height at current time steps position in p and z coordina *
* pvold,pvnew [Ks-1Pa-1] potential vorticity at old and new position           *
* qqold,qqnew        specific humidity at old and new position                 *
* thold,thnew        potential temperature at old and new position             *
* trerroru, trerrorv, trerrorw   memory of random errors added at last timestep*
* unew,vnew,wnew     Wind components at next time steps position of trajectory *
* uold,vold,wold     Wind components at current position of trajectory         *
* vvmax              maximum of u and v, help variable                         *
* wheight(nwzmax)    Heights of the model layers for w component               *
* xhelp,yhelp,zhelp  help variables                                            *
* xnew,ynew,znew     Next time step's spatial position of trajectory           *
* xold,yold,zold     Actual spatial position of trajectory                     *
* zdist              vertical grid distance at current position of trajectory  *
*                                                                              *
* Constants:                                                                   *
* cfl                factor, by which the time step has to be smaller than the *
*                    spatial CFL-criterion                                     *
* cflt               factor, by which the time step has to be smaller than the *
*                    temporal CFL-criterion                                    *
* deltahormax        maximum horizontal distance between two iterations        *
* deltavermax        maximum vertical distance between two iterations          *
* eps1               tiny number                                               *
* itermax            maximum number of iterations                              *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer i,j,nt,itime,nstop,idiff,indwz,ntstep,ntstep1,ntstep2
      integer ntstep3,minstep,lkind,lkindz,idummy,ngrid
      real xold,yold,zold,xnew,ynew,znew,xhelp,yhelp,zhelp,uint,vint
      real unew,vnew,wnew,uold,vold,wold,vvmax,zdist,levconst
      real pold,hold,pnew,hnew,pvold,pvnew,thold,thnew,qqold,qqnew
      real gasdev,xlon,ylat,xpol,ypol,gridsize
      real cgszll,xpolold,ypolold,corr,trerroru,trerrorv,trerrorw
      logical init,unc
      save idummy
      data idummy/-7/

      nstop=0
      ntstep=0

C Determine whether lat/long grid or polarstereographic projection
C is to be used
C Furthermore, determine which nesting level to be used
******************************************************************
c     write (*,*) 'petters',nglobal,yold,switchnorthg
      if (nglobal.and.(yold.gt.switchnorthg)) then
        ngrid=-1
      else if (sglobal.and.(yold.lt.switchsouthg)) then
        ngrid=-2
      else
        ngrid=0
        do 12 j=numbnests,1,-1
          if ((xold.gt.xln(j)).and.(xold.lt.xrn(j)).and.
     +    (yold.gt.yln(j)).and.(yold.lt.yrn(j))) then
            ngrid=j
            goto 13
          endif
12        continue
13      continue
      endif


C Get wind data at the current trajectory position
**************************************************

      call getwind(.true.,init,itime,levconst,xold,yold,zold,lkind,
     +lkindz,2,ngrid,uint,vint,wold,idiff,indwz,nstop)

      if (nstop.gt.1) return

C Transformation m/s --> grid units/time unit
*********************************************

      if (ngrid.ge.0) then          ! mother domain
        call utransform(uint,yold,uold)
        call vtransform(vint,vold)
      else if (ngrid.eq.-1) then    ! around north pole
        xlon=xlon0+xold*dx
        ylat=ylat0+yold*dy
        gridsize=1000.*cgszll(northpolemap,ylat,xlon)
        uold=uint/gridsize
        vold=vint/gridsize
      else if (ngrid.eq.-2) then    ! around south pole
        xlon=xlon0+xold*dx
        ylat=ylat0+yold*dy
        gridsize=1000.*cgszll(southpolemap,ylat,xlon)
        uold=uint/gridsize
        vold=vint/gridsize
      endif



C Calculate the time step determined by the CFL criterion
C Three independent CFL-criteria are used: 
C 1) In horizontal direction = displacement smaller than grid distance
C 2) In vertical direction = displacement smaller than grid distance
C 3) In temporal direction = distance between the 2 windfields used for interpol
C The final time step is the minimum of 1,2 and 3
********************************************************************************

      if (ngrid.lt.0) then
        ntstep1=999999
      else
      vvmax=max(abs(uold)*xresoln(ngrid),abs(vold)*yresoln(ngrid),eps1)
      ntstep1=int(min(1./vvmax/cfl,999999.))    ! 1
      endif

      if (lkind.eq.1) then
        zdist=wheight(indwz+1)-wheight(indwz)
        if (abs(wold).lt.eps1) wold=eps1
        ntstep2=int(min(zdist/abs(wold)/cfl,999999.)) ! 2
      else
        ntstep2=999999
      endif

      ntstep3=int(float(idiff)/cflt)                  ! 3

      ntstep=min(ntstep1,ntstep2,ntstep3)

      if (ntstep.lt.1) ntstep=1                       ! minimum step = 1 second


C Check, if such a short time step is possible. Under the assumption
C that all following time steps have the same length, check if this
C would lead to an exceedance of field dimensions. If this is the case
C increase the time step. However the time step cannot be greater than
C the time difference between 2 wind fields
**********************************************************************

      if ((inter.ne.1).and.(ntstep.lt.minstep)) then             
        ntstep=min(minstep,idiff)
        write(*,*) 'Warning: time step exceeds CFL-criterion!'
      endif

      if (ldirect.eq.-1) ntstep=-1*ntstep            ! backward t. -> negative


C If the last step of trajectory is reached, give nstop code 1
C Reduce the time step, so that trajectory ends exactly after lentra seconds
****************************************************************************

      if (abs(nt+ntstep).ge.abs(lentra)) then
        ntstep=ldirect*(abs(lentra-nt))
        nstop=1
      endif


C Calculate first guess position at time step nt+ntstep
*******************************************************

      if (ngrid.ge.0) then          ! mother domain
        xnew=xold+uold*float(ntstep)
        ynew=yold+vold*float(ntstep)
      else if (ngrid.eq.-1) then    ! around north pole
        call cll2xy(northpolemap,ylat,xlon,xpolold,ypolold)
        xpol=xpolold+uold*float(ntstep)
        ypol=ypolold+vold*float(ntstep)
        call cxy2ll(northpolemap,xpol,ypol,ylat,xlon)
        xnew=(xlon-xlon0)/dx
        ynew=(ylat-ylat0)/dy
      else if (ngrid.eq.-2) then    ! around south pole
        call cll2xy(southpolemap,ylat,xlon,xpolold,ypolold)
        xpol=xpolold+uold*float(ntstep)
        ypol=ypolold+vold*float(ntstep)
        call cxy2ll(southpolemap,xpol,ypol,ylat,xlon)
        xnew=(xlon-xlon0)/dx
        ynew=(ylat-ylat0)/dy
      endif
      if(lkind.eq.1) then
        znew=zold+wold*float(ntstep)
      else
        znew=zold
      endif


C Check position: If trajectory outside model domain, terminate it
******************************************************************

      if (znew.gt.heightmax) znew=heightmax
      if (znew.lt.heightmin) znew=heightmin


C If global data are available, use cyclic boundary condition
*************************************************************

      if (xglobal) then
        if (xnew.gt.float(nx-1)) xnew=xnew-float(nx-1)
        if (xnew.lt.0.) xnew=xnew+float(nx-1)
      endif


      if ((xnew.lt.0.).or.(xnew.gt.float(nx-1)).or.(ynew.lt.0.).or.
     +(ynew.gt.float(ny-1))) then
        nstop=2
        return
      endif


C Iteration. A maximum of itermax iterations is allowed. Iteration is
C terminated, when distance between two iterations is less than the maximum
C allowable distance deltadistmax
***************************************************************************

      do 10 i=1,itermax
        xhelp=xnew
        yhelp=ynew
        zhelp=znew

C Check whether position is within a nest
*****************************************

        if (ngrid.ge.0) then
          ngrid=0
          do 22 j=numbnests,1,-1
            if ((xnew.gt.xln(j)).and.(xnew.lt.xrn(j)).and.
     +      (ynew.gt.yln(j)).and.(ynew.lt.yrn(j))) then
              ngrid=j
              goto 23
            endif
22          continue
23        continue
        endif

C Get wind data at trajectory position at time step nt+ntstep
*************************************************************

        call getwind(.false.,init,itime+ntstep,levconst,xnew,ynew,
     +  znew,lkind,lkindz,i,ngrid,uint,vint,wnew,idiff,indwz,nstop)

        if (nstop.gt.1) return

C Transformation m/s --> grid units/time unit and advection
***********************************************************

        if (ngrid.ge.0) then
          call utransform(uint,ynew,unew)
          call vtransform(vint,vnew)
          xnew=xold+(uold+unew)*float(ntstep)/2.
          ynew=yold+(vold+vnew)*float(ntstep)/2.
        else if (ngrid.eq.-1) then    ! around north pole
          xlon=xlon0+xnew*dx
          ylat=ylat0+ynew*dy
          gridsize=1000.*cgszll(northpolemap,ylat,xlon)
          unew=uint/gridsize
          vnew=vint/gridsize
          xpol=xpolold+(uold+unew)*float(ntstep)/2.
          ypol=ypolold+(vold+vnew)*float(ntstep)/2.
          call cxy2ll(northpolemap,xpol,ypol,ylat,xlon)
          xnew=(xlon-xlon0)/dx
          ynew=(ylat-ylat0)/dy
        else if (ngrid.eq.-2) then    ! around south pole
          xlon=xlon0+xnew*dx
          ylat=ylat0+ynew*dy
          gridsize=1000.*cgszll(southpolemap,ylat,xlon)
          unew=uint/gridsize
          vnew=vint/gridsize
          xpol=xpolold+(uold+unew)*float(ntstep)/2.
          ypol=ypolold+(vold+vnew)*float(ntstep)/2.
          call cxy2ll(southpolemap,xpol,ypol,ylat,xlon)
          xnew=(xlon-xlon0)/dx
          ynew=(ylat-ylat0)/dy
        endif

        if(lkind.eq.1) then
          znew=zold+(wold+wnew)*float(ntstep)/2.
        endif

C Check position: If trajectory outside model domain, terminate it
******************************************************************

        if (znew.gt.heightmax) znew=heightmax
        if (znew.lt.heightmin) znew=heightmin


C If global data are available, use cyclic boundary condition
*************************************************************

        if (xglobal) then
          if (xnew.gt.float(nx-1)) xnew=xnew-float(nx-1)
          if (xnew.lt.0.) xnew=xnew+float(nx-1)
        endif


        if ((xnew.lt.0.).or.(xnew.gt.float(nx-1)).or.(ynew.lt.0.).or.
     +  (ynew.gt.float(ny-1))) then
          nstop=2
          return
        endif



C If the distance between two iterations is less than the maximum allowable
C one (in vertical and horizontal direction), leave the loop.
***************************************************************************

        if (lkind.eq.1) then
          if ((sqrt((xhelp-xnew)*(xhelp-xnew)+(yhelp-ynew)*(yhelp-ynew))      
     +    .lt.deltahormax).and.
     +    (abs(zhelp-znew).lt.deltavermax*zdist)) goto 20
        else
          if (sqrt((xhelp-xnew)*(xhelp-xnew)+(yhelp-ynew)*(yhelp-ynew))      
     +    .lt.deltahormax) goto 20
        endif

10      continue

20    continue


C Calculate random displacement for uncertainty trajectories
C using a Langevin equation
C For this, an autocorrelation of wind errors is calculated.
C This is not done near the poles!
************************************************************

      if (unc.and.(ngrid.ge.0)) then
        corr=exp(-1./(relaxtime*max(cfl,cflt)))
        trerroru=corr*trerroru+gasdev(idummy)*epsu*sqrt(1.-corr**2)
        xnew=xnew+trerroru*(uold+unew)*float(ntstep)/2.
        trerrorv=corr*trerrorv+gasdev(idummy)*epsv*sqrt(1.-corr**2)
        ynew=ynew+trerrorv*(vold+vnew)*float(ntstep)/2.
        trerrorw=corr*trerrorw+gasdev(idummy)*epsw*sqrt(1.-corr**2)
        znew=znew+trerrorw*(wold+wnew)*float(ntstep)/2.
        if (xglobal) then
          if (xnew.gt.float(nx-1)) xnew=xnew-float(nx-1)
          if (xnew.lt.0.) xnew=xnew+float(nx-1)
        endif
        if ((xnew.lt.0.).or.(xnew.gt.float(nx-1)).or.(ynew.lt.0.).or.
     +  (ynew.gt.float(ny-1))) then
          nstop=2
          return
        endif
        if (znew.gt.heightmax) znew=heightmax
        if (znew.lt.heightmin) znew=heightmin
      endif



C Calculate vertical position of trajectory in pressure and z coordinates.
C Interpolate also additional meteo data (PV, THETA, Spec. Hum.) to
C trajectory position.
C This is always done for the new positions, only for the first position
C of the trajectory it is necessary to calculate the old positions.
*************************************************************************
       
      if (init) then
        call getheight(itime,xold,yold,zold,pold,hold)
        call getmet(itime,xold,yold,zold,qqold,pvold,thold)
      endif
      call getheight(itime+ntstep,xnew,ynew,znew,pnew,hnew)
      call getmet(itime+ntstep,xnew,ynew,znew,qqnew,pvnew,thnew)

      return
      end
