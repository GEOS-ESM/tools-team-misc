      subroutine getwind(firststep,init,itime,levconst,xt,yt,zt,
     +lkind,lkindz,iter,ngrid,uint,vint,dzdt,idiff,indwz,nstop)
C                            i      i    i       i     i  i  i
C       i      i    i     i    o    o    o     o     o     o
********************************************************************************
*                                                                              *
*  This subroutine reads the wind fields and interpolates the wind fields to   *
*  the current trajectory position.                                            *
*  One has to distinguish between different types of trajectories, eg. isobaric*
*  on model layers or 3-dim. Interpolation is done separately for each of these*
*  types.                                                                      *
*                                                                              *
*    Author: A. Stohl                                                          *
*                                                                              *
*    26 April 1994                                                             *
*                                                                              *
*     Update: 23 December 1998 (Use of global domain and nesting)              *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* uint,vint,dzdt     wind components in grid units per second                  *
* firststep          .true. for first iteration of petterssen                  *
* idiff [s]          Temporal distance between the windfields used for interpol*
* indwz              index of the model layer beneath current position of traj.*
* init               .true. for first time step of trajectory                  *
* iter               number of iteration step                                  *
* itime [s]          current temporal position                                 *
* levconst           height of trajectory in Pa, m or K, depending on type of t*
* lkind              kind of trajectory (e.g. isobaric, 3-dimensional)         *
* lkindz             unit of z coordinate (1:masl, 2:magl, 3:hPa)              *
* memtime(3) [s]     times of the wind fields in memory                        *
* ngrid              points towards which grid shall be used                   *
* nstop              =greater 0, if trajectory calculation is finished         *
* xt,yt,zt           coordinates position for which wind data shall be calculat*
*                                                                              *
* Constants:                                                                   *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer itime,idiff,indexf,indwz,nstop,lkind,lkindz,iter,ngrid
      real xt,yt,zt,levconst,uint,vint,dzdt
      logical firststep,init


C Update of the wind fields that are currently kept in memory.
C If a new wind field is necessary, read it.
**************************************************************

      call getfields(firststep,itime,indexf,idiff,nstop)

      if (nstop.ge.3) return       ! error has occurred

C Call interpolation of the different types of trajectories.
************************************************************
     
      if (lkind.eq.1) then         ! full 3-dimensional trajectories
        call inter3d(xt,yt,zt,indexf,itime,init,firststep,
     +  iter,lkindz,ngrid,uint,vint,dzdt,indwz)
      else if (lkind.eq.2) then    ! trajectories on model layers
        call intermod(xt,yt,zt,indexf,itime,init,firststep,
     +  iter,lkindz,ngrid,uint,vint)
        dzdt=0.
      else if (lkind.eq.3) then    ! mixing layer trajectories
        call intermix(xt,yt,zt,levconst,indexf,itime,iter,ngrid,
     +  uint,vint)
        dzdt=0.
      else if (lkind.eq.4) then    ! isobaric trajectories
        call interisobar(xt,yt,zt,levconst,indexf,itime,init,firststep,
     +  iter,lkindz,ngrid,uint,vint)
        dzdt=0.
      else if (lkind.eq.5) then    ! isentropic trajectories
        call interisentrop(xt,yt,zt,levconst,indexf,itime,init,
     +  firststep,iter,lkindz,ngrid,uint,vint)
      endif

      return
      end
