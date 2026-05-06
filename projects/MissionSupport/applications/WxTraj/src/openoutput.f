      subroutine openoutput(error)
C                             o
********************************************************************************
*                                                                              *
*   This routine opens the trajectory data output files.                       *
*                                                                              *
*     Authors: A. Stohl                                                        *
*                                                                              *
*     16 February 1994                                                         *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* compoint(maxpoint)   comment for each startpoint                             *
* datestring           date and time of model run                              *
* error                .true., if error ocurred in subprogram, else .false.    *
* numpoint             number of starting points                               *
*                                                                              *
* Constants:                                                                   *
* unittraj             unittraj+i are connected to trajectory output (point i) *
* unittraji            unittraji+i are connected to interpolated trajectories  *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer i,j,k,l
      logical error
      character datestring*24
     

      error=.false.


C Check, if output file names are unequivocal
*********************************************

      do 1 i=1,numpoint-1
        do 1 j=i+1,numpoint
          if (compoint(i)(1:20).eq.compoint(j)(1:20).and.
     +    (compoint(i)(41:45).eq.compoint(j)(41:45))) then
            write(*,*) 'ERROR: 2 STARTING POINTS HAVE IDENTICAL NAMES.'
            write(*,*) 'CHANGE ONE NAME: ',compoint(i)
            stop
          endif
1         continue


C Look for runtime
******************

C      call fdate(datestring)


C If possible, use the first 20 characters of the comment for the file 
C name identification.
*********************************************************************

C 1. if wanted, original trajectories (flexible time step)
**********************************************************

      if ((inter.eq.0).or.(inter.eq.2)) then
      do 10 i=1,numpoint
        k=index(compoint(i),' ')-1
        k=min(k,20)                        
        if (compoint(i)(41:41).ne.'U') then       ! no uncertainty trajectory
          open(unittraj+i,file=path(2)(1:len(2))//'T_'//compoint(i)
     +    (1:k),status='new',err=998)         
        else                                      ! uncertainty trajectory
          open(unittraj+i,file=path(2)(1:len(2))//'T_'//compoint(i)
     +(1:k)//'_U'//compoint(i)(42:45),status='new',err=998)          
        endif
        write(unittraj+i,'(i8,a)') 42+4*numbnests,
     +  '                                       Number of header lines'
        write(unittraj+i,'(a69)') '*************************************
     +********************************'
        write(unittraj+i,'(a69)') '*                                           
     +                               *'
        write(unittraj+i,'(a69)') '*               FLEXTRA V4.0  MODEL O         
     +UTPUT                          *'
        write(unittraj+i,'(a69)') '*                     FOR ECMWF WINDF
     +IELDS                          *'
        write(unittraj+i,'(a69)') '*                                           
     +                               *'
        write(unittraj+i,'(a69)') '*************************************
     +********************************'
        write(unittraj+i,'(a69)') '*                                           
     +                               *'
        write(unittraj+i,'(a,a,a)') '*            TIME OF COMPUTATION: 
     + ' ,datestring(1:24),'            *'
        write(unittraj+i,'(a69)') '*                                           
     +                               *'
        write(unittraj+i,'(a69)') '*************************************
     +********************************'
        write(unittraj+i,'(a69)') '*                                           
     +                               *'
        if (kind(i).eq.1) then
        write(unittraj+i,'(a69)') '*           TYPE OF TRAJECTORIES:  3-
     +DIMENSIONAL                    *'
        else if (kind(i).eq.2) then
        write(unittraj+i,'(a69)') '*           TYPE OF TRAJECTORIES:  ON
     + MODEL LAYERS                  *'
        else if (kind(i).eq.3) then
        write(unittraj+i,'(a69)') '*           TYPE OF TRAJECTORIES:  MI
     +XED LAYER                      *'
        else if (kind(i).eq.4) then
        write(unittraj+i,'(a69)') '*           TYPE OF TRAJECTORIES:  IS
     +OBARIC                         *'
        else if (kind(i).eq.5) then
        write(unittraj+i,'(a69)') '*           TYPE OF TRAJECTORIES:  IS
     +ENTROPIC                       *'
        endif
        write(unittraj+i,'(a69)') '*                                           
     +                               *'
        write(unittraj+i,'(a69)') '*************************************
     +********************************'
        write(unittraj+i,'(a69)') '*                                           
     +                               *'
        write(unittraj+i,'(a69)') '*             INTEGRATION SCHEME:  PE          
     +TTERSSEN                       *'
        if (inpolkind.eq.1) then
        write(unittraj+i,'(a69)') '*           INTERPOLATION METHOD:  ID
     +EAL                            *'
        else
        write(unittraj+i,'(a69)') '*           INTERPOLATION METHOD:  LI
     +NEAR                           *'
        endif
        write(unittraj+i,'(a69)') '*                                           
     +                               *'
        write(unittraj+i,'(a,f5.2,a)') '*          SPATIAL CFL CRITERION
     +: ',cfl,'                             *'
        write(unittraj+i,'(a,f5.2,a)') '*         TEMPORAL CFL CRITERION
     +: ',cflt,'                             *'
        write(unittraj+i,'(a69)') '*                                           
     +                               *'
        write(unittraj+i,'(a69)') '*************************************
     +********************************'
        write(unittraj+i,'(a69)') '*                                           
     +                               *'
        write(unittraj+i,'(a)') '* START POINT COMMENT: '//
     +  compoint(i)(1:40)//'     *'
        write(unittraj+i,'(a)') '* MODEL RUN COMMENT: '//
     +  runcomment(1:47)//'*'
        write(unittraj+i,'(a69)') '*                                           
     +                               *'
        write(unittraj+i,'(a69)') '*************************************
     +********************************'
        write(unittraj+i,'(a69)') '*                                           
     +                               *'
        write(unittraj+i,'(a69)') '* INFORMATION ON WIND FIELDS USED FOR
     + COMPUTATIONS:                 *'
        write(unittraj+i,'(a69)') '*                                           
     +                               *'
        write(unittraj+i,'(a,i6,a)') '* NORMAL INTERVAL BETWEEN WIND FIE
     +LDS: ',idiffnorm,' SECONDS               *'
        write(unittraj+i,'(a,i6,a)') '* MAXIMUM INTERVAL BETWEEN WIND FI
     +ELDS: ',idiffmax,' SECONDS              *'
        write(unittraj+i,'(a69)') '*                                           
     +                               *'
        write(unittraj+i,'(a29,2i4,a32)') '* NUMBER OF VERTICAL LEVELS: 
     +  ',nuvz,nwz,'                               *'
        write(unittraj+i,'(a69)') '*                                           
     +                               *'
        write(unittraj+i,'(a69)') '* MOTHER DOMAIN:                            
     +                               *'
        write(unittraj+i,'(a,f7.2,a,f7.2,a,f6.2,a)')
     +  '* LONGITUDE RANGE: ',xlon0, ' TO ',xlon0+(nx-1)*dx,
     +  '   GRID DISTANCE: ',dx,'       *'
        write(unittraj+i,'(a,f7.2,a,f7.2,a,f6.2,a)')
     +  '* LATITUDE RANGE:  ',ylat0, ' TO ',ylat0+(ny-1)*dy,
     +  '   GRID DISTANCE: ',dy,'       *'

      do 300 l=1,numbnests
        write(unittraj+i,'(a69)') '*                                           
     +                               *'
        write(unittraj+i,'(a,i2,a)') '* NESTED DOMAIN NUMBER: ',l,'                       
     +                                     *'
        write(unittraj+i,'(a,f7.2,a,f7.2,a,f6.2,a)')
     +  '* LONGITUDE RANGE: ',xlon0n(l), ' TO ',
     +  xlon0n(l)+(nxn(l)-1)*dxn(l),
     +  '   GRID DISTANCE: ',dxn(l),'       *'
        write(unittraj+i,'(a,f7.2,a,f7.2,a,f6.2,a)')
     +  '* LATITUDE RANGE:  ',ylat0n(l), ' TO ',
     +  ylat0n(l)+(nyn(l)-1)*dyn(l),
     +  '   GRID DISTANCE: ',dyn(l),'       *'
300     continue

        write(unittraj+i,'(a69)') '*                                           
     +                               *'
        write(unittraj+i,'(a69)') '*************************************
     +********************************'
        write(unittraj+i,*)
10      continue
      endif


C 2. if wanted, interpolated trajectories (constant time step)
**************************************************************

      if (inter.ge.1) then
      do 20 i=1,numpoint
        k=index(compoint(i),' ')-1
        k=min(k,40)                        
        if (compoint(i)(41:41).ne.'U') then       ! no uncertainty trajectory
          open(unittraji+i,file=path(2)(1:len(2))//'TI_'//compoint(i)
     +    (1:k),status='new',err=999)         
        else                                      ! uncertainty trajectory
          open(unittraji+i,file=path(2)(1:len(2))//'TI_'//compoint(i)
     +(1:k)//'_U'//compoint(i)(42:45),status='new',err=999)          
        endif
        write(unittraji+i,'(i8,a)') 42+4*numbnests,
     +  '                                       Number of header lines'
       write(unittraji+i,'(a69)') '*************************************
     +********************************'
       write(unittraji+i,'(a69)') '*                                           
     +                               *'
       write(unittraji+i,'(a69)') '*                     FLEXTRA MODEL O         
     +UTPUT                          *'
       write(unittraji+i,'(a69)') '*                     FOR ECMWF WINDF
     +IELDS                          *'
       write(unittraji+i,'(a69)') '*                                           
     +                               *'
       write(unittraji+i,'(a69)') '*************************************
     +********************************'
       write(unittraji+i,'(a69)') '*                                           
     +                               *'
       write(unittraji+i,'(a,a,a)') '*            TIME OF COMPUTATION: 
     + ',datestring(1:24),'          *'
       write(unittraji+i,'(a69)') '*                                           
     +                               *'
       write(unittraji+i,'(a69)') '*************************************
     +********************************'
       write(unittraji+i,'(a69)') '*                                           
     +                               *'
        if (kind(i).eq.1) then
       write(unittraji+i,'(a69)') '*           TYPE OF TRAJECTORIES:  3-
     +DIMENSIONAL                    *'
        else if (kind(i).eq.2) then
       write(unittraji+i,'(a69)') '*           TYPE OF TRAJECTORIES:  ON
     + MODEL LAYERS                  *'
        else if (kind(i).eq.3) then
       write(unittraji+i,'(a69)') '*           TYPE OF TRAJECTORIES:  MI
     +XED LAYER                      *'
        else if (kind(i).eq.4) then
       write(unittraji+i,'(a69)') '*           TYPE OF TRAJECTORIES:  IS
     +OBARIC                         *'
        else if (kind(i).eq.5) then
       write(unittraji+i,'(a69)') '*           TYPE OF TRAJECTORIES:  IS
     +ENTROPIC                       *'
        endif
       write(unittraji+i,'(a69)') '*                                           
     +                               *'
       write(unittraji+i,'(a69)') '*************************************
     +********************************'
       write(unittraji+i,'(a69)') '*                                           
     +                               *'
       write(unittraji+i,'(a69)') '*             INTEGRATION SCHEME:  PE          
     +TTERSSEN                       *'
       if (inpolkind.eq.1) then
       write(unittraji+i,'(a69)') '*           INTERPOLATION METHOD:  ID
     +EAL                            *'
       else
       write(unittraji+i,'(a69)') '*           INTERPOLATION METHOD:  LI
     +NEAR                           *'
       endif
       write(unittraji+i,'(a69)') '*                                           
     +                               *'
       write(unittraji+i,'(a,f5.2,a)') '*          SPATIAL CFL CRITERION
     +: ',cfl,'                             *'
       write(unittraji+i,'(a,f5.2,a)') '*         TEMPORAL CFL CRITERION
     +: ',cflt,'                             *'
       write(unittraji+i,'(a69)') '*                                           
     +                               *'
       write(unittraji+i,'(a69)') '*************************************
     +********************************'
       write(unittraji+i,'(a69)') '*                                           
     +                               *'
       write(unittraji+i,'(a)') '* START POINT COMMENT: '//
     + compoint(i)(1:40)//'     *'
        write(unittraji+i,'(a)') '* MODEL RUN COMMENT: '//
     +  runcomment(1:47)//'*'
       write(unittraji+i,'(a69)') '*                                           
     +                               *'
       write(unittraji+i,'(a69)') '*************************************
     +********************************'
       write(unittraji+i,'(a69)') '*                                           
     +                               *'
       write(unittraji+i,'(a69)') '* INFORMATION ON WIND FIELDS USED FOR
     + COMPUTATIONS:                 *'
       write(unittraji+i,'(a69)') '*                                           
     +                               *'
       write(unittraji+i,'(a,i6,a)') '* NORMAL INTERVAL BETWEEN WIND FIE
     +LDS: ',idiffnorm,' SECONDS               *'
       write(unittraji+i,'(a,i6,a)') '* MAXIMUM INTERVAL BETWEEN WIND FI
     +ELDS: ',idiffmax,' SECONDS              *'
       write(unittraji+i,'(a69)') '*                                           
     +                               *'
       write(unittraji+i,'(a29,2i4,a32)') '* NUMBER OF VERTICAL LEVELS: 
     +  ',nuvz,nwz,'                               *'
       write(unittraji+i,'(a69)') '*                                           
     +                               *'
       write(unittraji+i,'(a69)') '* MOTHER DOMAIN:                            
     +                               *'
       write(unittraji+i,'(a,f7.2,a,f7.2,a,f6.2,a)')
     +  '* LONGITUDE RANGE: ',xlon0, ' TO ',xlon0+(nx-1)*dx,
     +  '   GRID DISTANCE: ',dx,'       *'
       write(unittraji+i,'(a,f7.2,a,f7.2,a,f6.2,a)')
     +  '* LATITUDE RANGE:  ',ylat0, ' TO ',ylat0+(ny-1)*dy,
     +  '   GRID DISTANCE: ',dy,'       *'

      do 400 l=1,numbnests
       write(unittraji+i,'(a69)') '*                                           
     +                               *'
       write(unittraji+i,'(a,i2,a)') '* NESTED DOMAIN NUMBER: ',l,'                       
     +                                     *'
       write(unittraji+i,'(a,f7.2,a,f7.2,a,f6.2,a)')
     +  '* LONGITUDE RANGE: ',xlon0n(l), ' TO ',
     +  xlon0n(l)+(nxn(l)-1)*dxn(l),
     +  '   GRID DISTANCE: ',dxn(l),'       *'
       write(unittraji+i,'(a,f7.2,a,f7.2,a,f6.2,a)')
     +  '* LATITUDE RANGE:  ',ylat0n(l), ' TO ',
     +  ylat0n(l)+(nyn(l)-1)*dyn(l),
     +  '   GRID DISTANCE: ',dyn(l),'       *'
400     continue

       write(unittraji+i,'(a69)') '*                                           
     +                               *'
       write(unittraji+i,'(a69)') '*************************************
     +********************************'
       write(unittraji+i,*)
20     continue
      endif

      return    

998   write(*,*)
      write(*,*) ' #### TRAJECTORY MODEL ERROR!   THE FILE       #### '
      write(*,*)
      write(*,*) '      '//path(2)(1:len(2))//'T_'//compoint(i)(1:k)
      write(*,*)
      write(*,*) ' #### (OR THIS FILE WITH A "_U" AT THE END)    #### '
      write(*,*) ' #### CANNOT BE OPENED. IF A FILE WITH THIS    #### '
      write(*,*) ' #### NAME ALREADY EXISTS, DELETE IT AND START #### '
      write(*,*) ' #### THE PROGRAM AGAIN.                       #### '
      error=.true.
      return

999   write(*,*)
      write(*,*) ' #### TRAJECTORY MODEL ERROR!   THE FILE       #### '
      write(*,*)
      write(*,*) '      '//path(2)(1:len(2))//'TI_'//compoint(i)(1:k)
      write(*,*)
      write(*,*) ' #### (OR THIS FILE WITH A "_U" AT THE END)    #### '
      write(*,*) ' #### CANNOT BE OPENED. IF A FILE WITH THIS    #### '
      write(*,*) ' #### NAME ALREADY EXISTS, DELETE IT AND START #### '
      write(*,*) ' #### THE PROGRAM AGAIN.                       #### '
      error=.true.

      return
      end
