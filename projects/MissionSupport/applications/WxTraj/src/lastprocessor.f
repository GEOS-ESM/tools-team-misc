      subroutine lastprocessor()
********************************************************************************
*                                                                              *
*   Back trajectories are produced in the wrong temporal direction by the      *
*   trajectory model. This means the first trajectory is the last one in the   *
*   file and the last one is the first one.                                    *
*   This routine reverses the direction so that the first date is also first   *
*   in the file.                                                               *
*                                                                              *
*   Author: A. Stohl                                                           *
*                                                                              *
*   8 April 1994                                                               *
*                                                                              *
********************************************************************************
*                                                                              *
* Variables:                                                                   *
* data(maximum)       data content of file                                     *
* header              header of file                                           *
* inter               index, if trajectory is interpolated or not              *
* last,numbmax        help variables                                           *
* numpoint            number of starting positions = number of output files    *
*                                                                              *
* Constants:                                                                   *
* maximum             maximum number of lines the procedure can handle         *
*                                                                              *
********************************************************************************

      include 'includepar'
      include 'includecom'

      integer maximum,numbmax,last,i,j,k
      parameter(maximum=50000)
      character header(98)*70,data(maximum)*80



C First: original trajectories (flexible time step)
*************************************************** 

      if ((inter.eq.0).or.(inter.eq.2)) then

C Reverse direction only for back trajectories.
C Read whole output file and store it in memory.
************************************************

      if (ldirect.eq.-1) then
        do 10 i=1,numpoint               ! loop over all starting points=files
          rewind(unittraj+i)
          do 20 j=1,42+4*numbnests
20          read(unittraj+i,'(a)') header(j)
          do 30 j=1,maximum
30          read(unittraj+i,'(a)',end=35) data(j)

35        numbmax=j-1


C If content of file exceeds available memory, don't reverse file
*****************************************************************

          if (numbmax.eq.maximum) then
            k=index(compoint(i),' ')-1
            k=min(k,7)                        
            write(*,*) '!!! Warning !!!  The content of file:'
            if (compoint(i)(41:41).ne.'U') then     ! no uncertainty trajectory
              write(*,*) 'T_'//compoint(i)(1:k)
            else
              write(*,*) 'T_'//compoint(i)(1:k)//'_U'//compoint(i)
     +        (42:45)
            endif
            write(*,*) 'is too long for postprocessing.'
            write(*,*) 'Please use manual postprocessor!'
            write(*,*)
            goto 10
          endif

C Rewind file and write it new with reversed direction
******************************************************

          rewind(unittraj+i)
          do 40 j=1,42+4*numbnests
40          write(unittraj+i,'(a)') header(j)
          last=numbmax
          do 50 j=numbmax,1,-1
            if (data(j)(1:1).eq.'D') then
              write(unittraj+i,'(a)') data(j)
              do 60 k=j+1,last
60              write(unittraj+i,'(a)') data(k)
              last=j-1
            endif
50          continue
10        continue

      endif

C Close output files
********************

      do 70 i=1,numpoint
70      close(unittraj+i)
      endif



C Second: interpolated trajectories (constant time step)
********************************************************

      if (inter.ge.1) then

C Reverse direction only for back trajectories.
C Read whole output file and store it in memory.
************************************************

      if (ldirect.eq.-1) then
        do 110 i=1,numpoint               ! loop over all starting points=files
          rewind(unittraji+i)
          do 120 j=1,42+4*numbnests
120         read(unittraji+i,'(a)') header(j)
          do 130 j=1,maximum
130         read(unittraji+i,'(a)',end=135) data(j)

135       numbmax=j-1


C If content of file exceeds available memory, don't reverse file
*****************************************************************

          if (numbmax.eq.maximum) then
            k=index(compoint(i),' ')-1
            k=min(k,7)                        
            write(*,*) '!!! Warning !!!  The content of file:'
            if (compoint(i)(41:41).ne.'U') then     ! no uncertainty trajectory
              write(*,*) 'TI_'//compoint(i)(1:k)
            else
              write(*,*) 'TI_'//compoint(i)(1:k)//'_U'//compoint(i)
     +        (42:45)
            endif
            write(*,*) 'is too long for postprocessing.'
            write(*,*) 'Please use manual postprocessor!'
            write(*,*)
            goto 110
          endif


C Rewind file and write it new with reversed direction
******************************************************

          rewind(unittraji+i)
          do 140 j=1,42+4*numbnests
140         write(unittraji+i,'(a)') header(j)
          last=numbmax
          do 150 j=numbmax,1,-1
            if (data(j)(1:1).eq.'D') then
              write(unittraji+i,'(a)') data(j)
              do 160 k=j+1,last
160             write(unittraji+i,'(a)') data(k)
              last=j-1
            endif
150         continue
110       continue

      endif


C Close output files and go for a coffee
****************************************

      do 170 i=1,numpoint
170     close(unittraji+i)
      endif



      return
      end
