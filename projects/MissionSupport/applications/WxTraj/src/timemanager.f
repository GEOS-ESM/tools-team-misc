      subroutine timemanager()                                  

********************************************************************************
*                                                                              *
* Handles the computation of trajectories, i.e. determines which               *
* trajectories have to be computed at what time.                               *
*                                                                              *
* Pointers (field numb(maxtra) are used to keep track of the trajectories.     *
* The pointers are kept sorted in the order of the next necessary time step,   *
* i.e., numb(1) points to the first trajectory that needs a time step.         *
* After doing the time step, the pointers are resorted, e.g., numb(2)          *
* changes to numb(1), numb(3) to numb(2), etc. Numb(1) is inserted between     *
* those two pointers, which need a time step before and after numb(1).         *
*                                                                              *
*                                                                              *
*     Author: A. Stohl                                                         *
*                                                                              *
*     6 February 1994                                                          *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* ideltas [s]        modelling period                                          *
* init               .true. for first time step of trajectory, false .else.    *
* interv [s]         interval between two trajectory starting times            *
* itime [s]          actual temporal position of calculation                   *
* ittra(maxtra,maxtime) temporal positions of trajectories                     *
* kind(maxtra)       type of trajectory (e.g. isobaric, 3-d, model layers,..)  *
* kindz(maxtra)      unit of height coordinate (1:masl, 2:magl, 3:hPa)         *
* ldirect            Temporal direction of trajectories (-1=backward,1=forward)*
* lentra             temporal length of one trajectory                         *
* levmem(maxtra) [K] height of isentropic trajectories                         *
* minstep            minimum time step (to prevent exceedance of dimensions)   *
* nextflight [s]     next time, initialization of FLIGHT trajectories is due   *
* npoint(maxtra)     index, which starting point the trajectory has            *
* nstop              = greater than 0 when trajectory calculation is finished  *
* ntstep [s]         time step of integration scheme                           *
* nttra(maxtra)      number of time steps which are already computed           *
* numb(maxtra)       pointer, which allows the identification of trajectories  *
* number             help variable                                             *
* numtra             actual number of trajectories in memory                   *
* xpoint(maxpoint), ypoint(maxpoint),zpoint(maxpoint) =                        *
* xpoint(maxpoint), ypoint(maxpoint),zpoint(maxpoint) =                        *
*                    starting positions of trajectories                        *
* xtra(maxtra,maxtime), ytra(maxtra,maxtime), ztra(maxtra,maxtime) =           *
*                    spatial positions of trajectories                         *
*                                                                              *
* Constants:                                                                   *
* maxtra             maximum number of trajectories                            *
* maxtime            maximum number of time steps                              *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer i,j,itime,ntstep,nstop(maxtra),lhelp,minstep,numb(maxtra)
      integer number,idummy,ldat,ltim,ninterv1,ninterv2,ninterv3
      real levmem(maxtra),gasdev
      logical init,unc,error
      double precision juldate
      save idummy

      data idummy/-7/


      do 5 i=1,maxtra
5       nttra(i)=0
      numtra=0                                      ! first: no trajectories


C Loop over the whole modelling period in time steps of seconds
***************************************************************

      do 10 itime=0,ideltas,ldirect
        if (mod(itime,3600).eq.0) write(*,*) itime,' SECONDS SIMULATED'


C Check whether some trajectories have already finished. If so, call output
C and make memory available for new trajectory.
***************************************************************************

        if (numtra.gt.0) then

          i=1
33        continue
          number=numb(i)
       
          if (abs(ittra(number,nttra(number))).gt.abs(itime)) goto 37
          if ((nstop(number).gt.0).and.               ! trajectory terminated
     +       (ittra(number,nttra(number)).eq.itime)) then     ! output time
c           if (modecet.ne.2) then                    ! don't make output for CETs
              call trajout(number,nstop(number))      ! output of trajectory
              nttra(number)=0                         ! reinitialize
              do 35 j=i,numtra-1                      ! resort the pointers
35              numb(j)=numb(j+1)
c           endif
            numtra=numtra-1                           ! forget one trajectory
            i=i-1
          endif
          if (i.lt.numtra) then
            i=i+1
            goto 33
          else
            goto 37 
          endif

37        continue
        endif



C Check whether new trajectories have to be started
C 1nd if: check if time within modeling period
C 2nd if: check if the interval interv has passed by (for NORMAL
C         trajectories) or if a FLIGHT traj. is due to be started
*****************************************************************

        if (abs(itime).le.abs(ideltas-lentra)) then
          if (modecet.eq.3) then       ! FLIGHT mode
41          continue
            if ((nextflight.eq.itime).and.(numtra.lt.maxtra)) then
              do 43 i=numtra,1,-1         ! shift pointers
43              numb(i+1)=numb(i)
              numtra=numtra+1             ! increase number of traj.

              j=1                         ! initialize FLIGHT traj.
42            continue
              if (nttra(j).eq.0) then
                numb(1)=j
                npoint(j)=1
                levmem(j)=zpoint(1)
                xtra(j,1)=xpoint(1)
                ytra(j,1)=ypoint(1)
                ztra(j,1)=zpoint(1)
                ittra(j,1)=itime
                nttra(j)=1
                kind(j)=kind(1)
                kindz(j)=kindz(1)
              else 
                j=j+1
                if (j.gt.maxtra) stop 'timemanager: too many trajectorie
     +s. Increase parameter maxtra and re-start'
                goto 42
              endif

C Read the next FLIGHT traj. 
****************************

              read(unitpoin,*,err=15,end=15) ldat,ltim
              read(unitpoin,*,err=15,end=15) xpoint(1)
              read(unitpoin,*,err=15,end=15) ypoint(1)
              read(unitpoin,*,err=15,end=15) zpoint(1)
              read(unitpoin,*,err=15,end=15)
              nextflight=nint(sngl(juldate(ldat,ltim)-bdate)*86400.)
              call coordtrafo(error)
              if (error) stop
              call subtractoro()
              if (kindz(1).eq.3) zpoint(1)=zpoint(1)*100.

              goto 41
            endif

          else if (mod(itime,interv).eq.0) then !non-FLIGHT mode, initialization due
            do 40 i=numtra,1,-1                       ! shift pointers
40            numb(i+numpoint)=numb(i)
            numtra=numtra+numpoint                    ! increase number of traj.


C Initialize new trajectories
*****************************

            j=1
            do 50 i=1,numpoint        ! search for available memory
55            continue
              if (nttra(j).eq.0) then
                numb(i)=j
                npoint(j)=i
                levmem(j)=zpoint(i)
                xtra(j,1)=xpoint(i)
                ytra(j,1)=ypoint(i)
                ztra(j,1)=zpoint(i)
                ittra(j,1)=itime
                nttra(j)=1
                kind(j)=kind(i)
                kindz(j)=kindz(i)
                randerroru(j)=gasdev(idummy)*epsu
                randerrorv(j)=gasdev(idummy)*epsv
                randerrorw(j)=gasdev(idummy)*epsw
              else 
                j=j+1
                goto 55
              endif
50            continue

          endif
        endif



C Check whether a time step of the first trajectory is necessary
****************************************************************

15      continue
        if (numtra.gt.0) then
        
        number=numb(1)

        if (ittra(number,nttra(number)).eq.itime) then

C minstep is the minimum possible time step. It is necessary to 
C guarantee that the sum of all time steps along the trajectory
C is smaller than maxtime (otherwise exceedance of dimensions).
***************************************************************

          lhelp=itime-ittra(number,1)         ! "age" of trajectory
          minstep=abs(lentra-lhelp)/(maxtime-nttra(number))+1


C Mark first time step of trajectory with variable init=.TRUE.
**************************************************************

          if (nttra(number).eq.1) then
            init=.true.
          else
            init=.false.
          endif


C Determine whether current trajectory is an uncertainty trajectory
*******************************************************************

          if (compoint(npoint(number))(41:41).eq.'U') then
            unc=.TRUE.
          else
            unc=.FALSE.
          endif
           

C Calculate the next step of the trajectory
*******************************************

          call petters(minstep,lhelp,itime,xtra(number,nttra(number)),
     +    ytra(number,nttra(number)),ztra(number,nttra(number)),
     +    ptra(number,nttra(number)),htra(number,nttra(number)),
     +    qqtra(number,nttra(number)),pvtra(number,nttra(number)),
     +    thtra(number,nttra(number)),
     +    xtra(number,nttra(number)+1),ytra(number,nttra(number)+1),
     +    ztra(number,nttra(number)+1),ptra(number,nttra(number)+1),
     +    htra(number,nttra(number)+1),qqtra(number,nttra(number)+1),
     +    pvtra(number,nttra(number)+1),thtra(number,nttra(number)+1),
     +    ntstep,nstop(number),
     +    levmem(number),init,kind(number),kindz(number),unc,
     +    randerroru(number),randerrorv(number),randerrorw(number))

          ittra(number,nttra(number)+1)=ittra(number,nttra(number))+
     +    ntstep

          nttra(number)=nttra(number)+1         ! one more time step


C If output is only due with constant time steps, disregard all points
C not needed for interpolation: delete previous time step
**********************************************************************

          if ((inter.eq.1).and.(nttra(number).ge.3)) then
            ninterv1=abs(ittra(number,nttra(number))-ittra(number,1))/
     +      interstep
            ninterv2=abs(ittra(number,nttra(number)-1)-ittra(number,1))/
     +      interstep
            ninterv3=abs(ittra(number,nttra(number)-2)-ittra(number,1))/
     +      interstep
            if ((ninterv1.eq.ninterv2).and.(ninterv1.eq.ninterv3)) then
              ittra(number,nttra(number)-1)=ittra(number,nttra(number))
              xtra(number,nttra(number)-1)=xtra(number,nttra(number))
              ytra(number,nttra(number)-1)=ytra(number,nttra(number))
              ztra(number,nttra(number)-1)=ztra(number,nttra(number))
              ptra(number,nttra(number)-1)=ptra(number,nttra(number))
              htra(number,nttra(number)-1)=htra(number,nttra(number))
              qqtra(number,nttra(number)-1)=qqtra(number,nttra(number))
              pvtra(number,nttra(number)-1)=pvtra(number,nttra(number))
              thtra(number,nttra(number)-1)=thtra(number,nttra(number))
              nttra(number)=nttra(number)-1
            endif
          endif

C If field dimension is reached, trajectory has to be terminated
****************************************************************

          if (nttra(number).eq.maxtime) nstop(number)=5


C If trajectory has ended, give the next time step the time of normal
C length of trajectory and wait for output of trajectory
*********************************************************************

          if (nstop(number).gt.0) then
            ittra(number,nttra(number))=ittra(number,1)+lentra
          endif


C Shift the trajectories in a way that the current trajectory can be
C inserted at the right position.
C The position is determined by the time, when the next integration step
C is necessary.
*************************************************************************

          do 20 i=1,numtra-1
            if (abs(ittra(numb(i+1),nttra(numb(i+1)))).lt.
     +      abs(ittra(number,nttra(number)))) then
              numb(i)=numb(i+1)
            else
              numb(i)=number                      ! insert the pointer
              goto 15                             ! leave loop
            endif
20          continue

C The next statement can only be reached, when the current trajectory is the
C last one that needs an integration step.
****************************************************************************

          numb(numtra)=number
          goto 15                                 ! check next trajectory
        endif
        endif


10     continue

      return
      end
