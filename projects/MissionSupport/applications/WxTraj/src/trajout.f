      subroutine trajout(numb,nstop)          
C                         i     i
********************************************************************************
*                                                                              *
*     This routine writes the output files.                                    *
*                                                                              *
*     Authors: A. Stohl                                                        *
*                                                                              *
*     2 February 1994                                                          *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* bdate                 beginning date of modelling period                     *
*                                                                              *
* idate,itime           date and time,help variables                           *
* inter                 output with flexible (0,2) or constant time step (1,2) *
* ldim                  number of interpolated time steps                      *
* maxnests              maximum number of nesting levels                       *
* ngrid                 nesting level to be used                               *
* npoint(maxtra)        number of starting point for each trajectory           *
* nstop                 error code                                             *
* nttra(maxtra)         number of time steps along the trajectory              *
* numb                  number of trajectory to be written to output file      *
* juldat                Julian starting date of trajectory                     *
* xtra,ytra,ztra(maxtra,maxtime)     grid coordinates of trajectory (flexible) *
* xtraint,ytraint,ztraint(maxitime)  grid coordinates of trajectory (flexible) *
*                                                                              *
* Constants:                                                                   *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer nstop,numb,idate,itime,i,j,ldimi,ngrid
      real orotra(maxtime),orotraint(maxitime),xtn,ytn
      double precision juldat


C If serious error has occurred, reduce the number of time steps by 1.
C This has to be done to exclude unrealistic values.
**********************************************************************

      if (nstop.gt.1) nttra(numb)=nttra(numb)-1


C Add the height of the orography to give height above sea level
****************************************************************

      do 10 i=1,nttra(numb)

C Determine which nesting level to be used
******************************************

        ngrid=0
        do 12 j=numbnests,1,-1
          if ((xtra(numb,i).gt.xln(j)).and.(xtra(numb,i).lt.xrn(j)).and.
     +    (ytra(numb,i).gt.yln(j)).and.(ytra(numb,i).lt.yrn(j))) then
            ngrid=j
            goto 13
          endif
12        continue
13      continue

        if (ngrid.eq.0) then
          call orolininterpol(oro,nxmax,nymax,nx,ny,xtra(numb,i),
     +    ytra(numb,i),orotra(i))
        else
          xtn=(xtra(numb,i)-xln(ngrid))*xresoln(ngrid)
          ytn=(ytra(numb,i)-yln(ngrid))*yresoln(ngrid)
          call orolininterpoln(oron,maxnests,nxmaxn,nymaxn,ngrid,
     +    nxn,nyn,xtn,ytn,orotra(i))
        endif
10    htra(numb,i)=htra(numb,i)+orotra(i)


C Conversion of grid coordinates to geografical coordinates
***********************************************************

      do 30 i=1,nttra(numb)
        call lamphi_ecmwf(xtra(numb,i),ytra(numb,i),xtra(numb,i),
     +  ytra(numb,i))
30      ptra(numb,i)=ptra(numb,i)/100.          ! conversion Pa -> hPa


C If trajectory information is wanted with constant time step, interpolate the
C trajectory to constant time step
******************************************************************************

      if (inter.ge.1) then
        call trajinterpol(numb,ldimi,orotra,orotraint)
      endif


C Calculate starting date and time of trajectory
************************************************

      juldat=bdate+dble(float(ittra(numb,1)))/86400.
      call caldate(juldat,idate,itime)


C Output of normal (non-CET) trajectories
*****************************************

      if (modecet.eq.1) then


C Output of trajectories with flexible time step
************************************************

      if ((inter.eq.0).or.(inter.eq.2)) then
        if (nttra(numb).eq.1) nttra(numb)=0
        write(unittraj+npoint(numb),'(a6,i8,a9,i8,a16,i1,a16,i5)') 
     +  'DATE: ',idate,'    TIME:',itime,'    STOP INDEX: ',nstop,
     +  '    # OF POINTS:',nttra(numb)
        write(unittraj+npoint(numb),67) '    SECS',
     +  '   LONGIT','   LATIT','    ETA','  PRESS','     Z',' Z-ORO',
     +'     PV',' THETA','       Q'
        do 40 i=1,nttra(numb)
40        write(unittraj+npoint(numb),66) 
     +    ittra(numb,i)-ittra(numb,1),xtra(numb,i),ytra(numb,i),
     +    ztra(numb,i)*zdirect,ptra(numb,i),htra(numb,i),
     +    htra(numb,i)-orotra(i),pvtra(numb,i),thtra(numb,i),
     +    qqtra(numb,i)
      endif


C Output of trajectories with constant time step
************************************************

      if (inter.ge.1) then                         
        if (ldimi.eq.1) ldimi=0
        write(unittraji+npoint(numb),'(a6,i8,a9,i8,a16,i1,a16,i5)') 
     +  'DATE: ',idate,'    TIME:',itime,'    STOP INDEX: ',nstop,
     +  '    # OF POINTS:',ldimi
        write(unittraji+npoint(numb),67) '    SECS',
     +  '   LONGIT','   LATIT','    ETA','  PRESS','     Z',' Z-ORO',
     +'     PV',' THETA','       Q'
        do 50 i=1,ldimi
50        write(unittraji+npoint(numb),66) 
     +    ittraint(i)-ittraint(1),xtraint(i),ytraint(i),ztraint(i)*
     +    zdirect,ptraint(i),htraint(i),htraint(i)-orotraint(i),
     +    pvtraint(i),thtraint(i),qqtraint(i)
      endif

C Output of CET or FLIGHT trajectories
**************************************

      else

C Output of trajectories with flexible time step
************************************************

      if ((inter.eq.0).or.(inter.eq.2)) then
        if (nttra(numb).eq.1) nttra(numb)=0
        write(unittraj,'(a6,i8,a9,i8,a16,i1,a16,i5)') 
     +  'DATE: ',idate,'    TIME:',itime,'    STOP INDEX: ',nstop,
     +  '    # OF POINTS:',nttra(numb)
        write(unittraj,67) '    SECS',
     +  '   LONGIT','   LATIT','    ETA','  PRESS','     Z',' Z-ORO',
     +'     PV',' THETA','       Q'
        do 140 i=1,nttra(numb)
140       write(unittraj,66) 
     +    ittra(numb,i)-ittra(numb,1),xtra(numb,i),ytra(numb,i),
     +    ztra(numb,i)*zdirect,ptra(numb,i),htra(numb,i),
     +    htra(numb,i)-orotra(i),pvtra(numb,i),thtra(numb,i),
     +    qqtra(numb,i)
      endif


C Output of trajectories with constant time step
************************************************

      if (inter.ge.1) then                         
        if (ldimi.eq.1) ldimi=0
        write(unittraji,'(a6,i8,a9,i8,a16,i1,a16,i5)') 
     +  'DATE: ',idate,'    TIME:',itime,'    STOP INDEX: ',nstop,
     +  '    # OF POINTS:',ldimi
        write(unittraji,67) '    SECS',
     +  '   LONGIT','   LATIT','    ETA','  PRESS','     Z',' Z-ORO',
     +'     PV',' THETA','       Q'
        do 150 i=1,ldimi
150       write(unittraji,66) 
     +    ittraint(i)-ittraint(1),xtraint(i),ytraint(i),ztraint(i)*
     +    zdirect,ptraint(i),htraint(i),htraint(i)-orotraint(i),
     +    pvtraint(i),thtraint(i),qqtraint(i)
      endif

      endif

66    format(i9,2f9.4,f7.4,f7.1,2f8.1,f8.3,f6.1,e9.2)
67    format(a9,a9,a9,a7,a7,a8,a8,a8,a6,a9)

      return    
      end
