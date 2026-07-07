      subroutine opencetoutput(error)
C                                o
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

      integer k,l
      logical error
      character datestring*24
     

      error=.false.


C Look for runtime
******************

C      call fdate(datestring)


C If possible, use the first 20 characters of the comment for the file 
C name identification.
*********************************************************************

C 1. if wanted, original trajectories (flexible time step)
**********************************************************

      if ((inter.eq.0).or.(inter.eq.2)) then
        k=index(compoint(1),' ')-1
        k=min(k,20)                        
          open(unittraj,file=path(2)(1:len(2))//'CET_'//compoint(1)
     +    (1:k),status='new',err=998)         
          write(unittraj,'(i8,a)') 42+4*numbnests,
     +  '                                       Number of header lines'
          write(unittraj,'(a69)') '*************************************
     +********************************'
          write(unittraj,'(a69)') '*                                           
     +                               *'
          write(unittraj,'(a69)') '*               FLEXTRA V4.0  MODEL O         
     +UTPUT                          *'
          write(unittraj,'(a69)') '*                     FOR ECMWF WINDF
     +IELDS                          *'
          write(unittraj,'(a69)') '*                                           
     +                               *'
          write(unittraj,'(a69)') '*************************************
     +********************************'
          write(unittraj,'(a69)') '*                                           
     +                               *'
          write(unittraj,'(a,a,a)') '*            TIME OF COMPUTATION: 
     + ' ,datestring(1:24),'            *'
          write(unittraj,'(a69)') '*                                           
     +                               *'
          write(unittraj,'(a69)') '*************************************
     +********************************'
          write(unittraj,'(a69)') '*                                           
     +                               *'
          if (kind(1).eq.1) then
          write(unittraj,'(a69)') '*           TYPE OF TRAJECTORIES:  3-
     +DIMENSIONAL                    *'
          else if (kind(1).eq.2) then
          write(unittraj,'(a69)') '*           TYPE OF TRAJECTORIES:  ON
     + MODEL LAYERS                  *'
          else if (kind(1).eq.4) then
          write(unittraj,'(a69)') '*           TYPE OF TRAJECTORIES:  IS
     +OBARIC                         *'
          else if (kind(1).eq.5) then
          write(unittraj,'(a69)') '*           TYPE OF TRAJECTORIES:  IS
     +ENTROPIC                       *'
          endif
          write(unittraj,'(a69)') '*                                           
     +                               *'
          write(unittraj,'(a69)') '*************************************
     +********************************'
          write(unittraj,'(a69)') '*                                           
     +                               *'
          write(unittraj,'(a69)') '*             INTEGRATION SCHEME:  PE          
     +TTERSSEN                       *'
          if (inpolkind.eq.1) then
          write(unittraj,'(a69)') '*           INTERPOLATION METHOD:  ID
     +EAL                            *'
          else
          write(unittraj,'(a69)') '*           INTERPOLATION METHOD:  LI
     +NEAR                           *'
          endif
          write(unittraj,'(a69)') '*                                           
     +                               *'
          write(unittraj,'(a,f5.2,a)') '*          SPATIAL CFL CRITERION
     +: ',cfl,'                             *'
          write(unittraj,'(a,f5.2,a)') '*         TEMPORAL CFL CRITERION
     +: ',cflt,'                             *'
          write(unittraj,'(a69)') '*                                           
     +                               *'
          write(unittraj,'(a69)') '*************************************
     +********************************'
          write(unittraj,'(a69)') '*                                           
     +                               *'
          write(unittraj,'(a)') '* START POINT COMMENT: '//
     +  compoint(1)(1:40)//'     *'
          write(unittraj,'(a)') '* MODEL RUN COMMENT: '//
     +  runcomment(1:47)//'*'
          write(unittraj,'(a69)') '*                                           
     +                               *'
          write(unittraj,'(a69)') '*************************************
     +********************************'
          write(unittraj,'(a69)') '*
     +                               *'
          write(unittraj,'(a69)') '* INFORMATION ON WIND FIELDS USED FOR
     + COMPUTATIONS:                 *'
          write(unittraj,'(a69)') '*
     +                               *'
          write(unittraj,'(a,i6,a)') '* NORMAL INTERVAL BETWEEN WIND FIE
     +LDS: ',idiffnorm,' SECONDS               *'
          write(unittraj,'(a,i6,a)') '* MAXIMUM INTERVAL BETWEEN WIND FI
     +ELDS: ',idiffmax,' SECONDS              *'
          write(unittraj,'(a69)') '*
     +                               *'
          write(unittraj,'(a29,2i4,a32)') '* NUMBER OF VERTICAL LEVELS:
     +  ',nuvz,nwz,'                               *'
          write(unittraj,'(a69)') '*
     +                               *'
          write(unittraj,'(a69)') '* MOTHER DOMAIN:
     +                               *'
          write(unittraj,'(a,f7.2,a,f7.2,a,f6.2,a)')
     +  '* LONGITUDE RANGE: ',xlon0, ' TO ',xlon0+(nx-1)*dx,
     +  '   GRID DISTANCE: ',dx,'       *'
          write(unittraj,'(a,f7.2,a,f7.2,a,f6.2,a)')
     +  '* LATITUDE RANGE:  ',ylat0, ' TO ',ylat0+(ny-1)*dy,
     +  '   GRID DISTANCE: ',dy,'       *'

      do 300 l=1,numbnests
          write(unittraj,'(a69)') '*
     +                               *'
          write(unittraj,'(a,i2,a)') '* NESTED DOMAIN NUMBER: ',l,'
     +                                     *'
          write(unittraj,'(a,f7.2,a,f7.2,a,f6.2,a)')
     +  '* LONGITUDE RANGE: ',xlon0n(l), ' TO ',
     +  xlon0n(l)+(nxn(l)-1)*dxn(l),
     +  '   GRID DISTANCE: ',dxn(l),'       *'
          write(unittraj,'(a,f7.2,a,f7.2,a,f6.2,a)')
     +  '* LATITUDE RANGE:  ',ylat0n(l), ' TO ',
     +  ylat0n(l)+(nyn(l)-1)*dyn(l),
     +  '   GRID DISTANCE: ',dyn(l),'       *'
300     continue

          write(unittraj,'(a69)') '*
     +                               *'
          write(unittraj,'(a69)') '*************************************
     +********************************'

          write(unittraj,*)
      endif


C 2. if wanted, interpolated trajectories (constant time step)
**************************************************************

      if (inter.ge.1) then
        k=index(compoint(1),' ')-1
        k=min(k,20)                        
        open(unittraji,file=path(2)(1:len(2))//'CETI_'//compoint(1)
     +  (1:k),status='new',err=999)         
         write(unittraji,'(i8,a)') 42+4*numbnests,
     +  '                                       Number of header lines'
         write(unittraji,'(a69)') '*************************************
     +********************************'
         write(unittraji,'(a69)') '*                                           
     +                               *'
         write(unittraji,'(a69)') '*                     FLEXTRA MODEL O         
     +UTPUT                          *'
         write(unittraji,'(a69)') '*                     FOR ECMWF WINDF
     +IELDS                          *'
         write(unittraji,'(a69)') '*                                           
     +                               *'
         write(unittraji,'(a69)') '*************************************
     +********************************'
         write(unittraji,'(a69)') '*                                           
     +                               *'
         write(unittraji,'(a,a,a)') '*            TIME OF COMPUTATION: 
     + ',datestring(1:24),'          *'
         write(unittraji,'(a69)') '*                                           
     +                               *'
         write(unittraji,'(a69)') '*************************************
     +********************************'
         write(unittraji,'(a69)') '*                                           
     +                               *'
         if (kind(1).eq.1) then
         write(unittraji,'(a69)') '*           TYPE OF TRAJECTORIES:  3-
     +DIMENSIONAL                    *'
         else if (kind(1).eq.2) then
         write(unittraji,'(a69)') '*           TYPE OF TRAJECTORIES:  ON
     + MODEL LAYERS                  *'
         else if (kind(1).eq.4) then
         write(unittraji,'(a69)') '*           TYPE OF TRAJECTORIES:  IS
     +OBARIC                         *'
         else if (kind(1).eq.5) then
         write(unittraji,'(a69)') '*           TYPE OF TRAJECTORIES:  IS
     +ENTROPIC                       *'
         endif
         write(unittraji,'(a69)') '*                                           
     +                               *'
         write(unittraji,'(a69)') '*************************************
     +********************************'
         write(unittraji,'(a69)') '*                                           
     +                               *'
         write(unittraji,'(a69)') '*             INTEGRATION SCHEME:  PE          
     +TTERSSEN                       *'
         if (inpolkind.eq.1) then
         write(unittraji,'(a69)') '*           INTERPOLATION METHOD:  ID
     +EAL                            *'
         else
         write(unittraji,'(a69)') '*           INTERPOLATION METHOD:  LI
     +NEAR                           *'
         endif
         write(unittraji,'(a69)') '*                                           
     +                               *'
         write(unittraji,'(a,f5.2,a)') '*          SPATIAL CFL CRITERION
     +: ',cfl,'                             *'
         write(unittraji,'(a,f5.2,a)') '*         TEMPORAL CFL CRITERION
     +: ',cflt,'                             *'
         write(unittraji,'(a69)') '*                                           
     +                               *'
         write(unittraji,'(a69)') '*************************************
     +********************************'
         write(unittraji,'(a69)') '*                                           
     +                               *'
         write(unittraji,'(a)') '* START POINT COMMENT: '//
     + compoint(1)(1:40)//'     *'
         write(unittraji,'(a)') '* MODEL RUN COMMENT: '//
     +  runcomment(1:47)//'*'
         write(unittraji,'(a69)') '*                                           
     +                               *'
         write(unittraji,'(a69)') '*************************************
     +********************************'
         write(unittraji,'(a69)') '*
     +                               *'
         write(unittraji,'(a69)') '* INFORMATION ON WIND FIELDS USED FOR
     + COMPUTATIONS:                 *'
         write(unittraji,'(a69)') '*
     +                               *'
         write(unittraji,'(a,i6,a)') '* NORMAL INTERVAL BETWEEN WIND FIE
     +LDS: ',idiffnorm,' SECONDS               *'
         write(unittraji,'(a,i6,a)') '* MAXIMUM INTERVAL BETWEEN WIND FI
     +ELDS: ',idiffmax,' SECONDS              *'
         write(unittraji,'(a69)') '*
     +                               *'
         write(unittraji,'(a29,2i4,a32)') '* NUMBER OF VERTICAL LEVELS:
     +  ',nuvz,nwz,'                               *'
         write(unittraji,'(a69)') '*
     +                               *'
         write(unittraji,'(a69)') '* MOTHER DOMAIN:
     +                               *'
         write(unittraji,'(a,f7.2,a,f7.2,a,f6.2,a)')
     +  '* LONGITUDE RANGE: ',xlon0, ' TO ',xlon0+(nx-1)*dx,
     +  '   GRID DISTANCE: ',dx,'       *'
         write(unittraji,'(a,f7.2,a,f7.2,a,f6.2,a)')
     +  '* LATITUDE RANGE:  ',ylat0, ' TO ',ylat0+(ny-1)*dy,
     +  '   GRID DISTANCE: ',dy,'       *'

      do 400 l=1,numbnests
         write(unittraji,'(a69)') '*
     +                               *'
         write(unittraji,'(a,i2,a)') '* NESTED DOMAIN NUMBER: ',l,'
     +                                     *'
         write(unittraji,'(a,f7.2,a,f7.2,a,f6.2,a)')
     +  '* LONGITUDE RANGE: ',xlon0n(l), ' TO ',
     +  xlon0n(l)+(nxn(l)-1)*dxn(l),
     +  '   GRID DISTANCE: ',dxn(l),'       *'
         write(unittraji,'(a,f7.2,a,f7.2,a,f6.2,a)')
     +  '* LATITUDE RANGE:  ',ylat0n(l), ' TO ',
     +  ylat0n(l)+(nyn(l)-1)*dyn(l),
     +  '   GRID DISTANCE: ',dyn(l),'       *'
400     continue

         write(unittraji,'(a69)') '*
     +                               *'
         write(unittraji,'(a69)') '*************************************
     +********************************'
         write(unittraji,*)
      endif

      return    

998   write(*,*)
      write(*,*) ' #### TRAJECTORY MODEL ERROR!   THE FILE       #### '
      write(*,*)
      write(*,*) '    '//path(2)(1:len(2))//'CET_'//compoint(1)(1:k)
      write(*,*)
      write(*,*) ' #### CANNOT BE OPENED. IF A FILE WITH THIS    #### '
      write(*,*) ' #### NAME ALREADY EXISTS, DELETE IT AND START #### '
      write(*,*) ' #### THE PROGRAM AGAIN.                       #### '
      error=.true.
      return

999   write(*,*)
      write(*,*) ' #### TRAJECTORY MODEL ERROR!   THE FILE       #### '
      write(*,*)
      write(*,*) '     '//path(2)(1:len(2))//'CETI_'//compoint(1)(1:k)
      write(*,*)
      write(*,*) ' #### CANNOT BE OPENED. IF A FILE WITH THIS    #### '
      write(*,*) ' #### NAME ALREADY EXISTS, DELETE IT AND START #### '
      write(*,*) ' #### THE PROGRAM AGAIN.                       #### '
      error=.true.

      return
      end
