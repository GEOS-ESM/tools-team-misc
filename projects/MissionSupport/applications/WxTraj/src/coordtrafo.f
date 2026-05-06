      subroutine coordtrafo(error)
***********************************************************************
*                                                                     * 
*             TRAJECTORY MODEL SUBROUTINE COORDTRAFO                  *
*                                                                     *
***********************************************************************
*                                                                     * 
*             AUTHOR:      G. WOTAWA                                  *
*             DATE:        1994-02-07                                 *
*             LAST UPDATE: ----------                                 *
*                                                                     * 
***********************************************************************
*                                                                     *
* DESCRIPTION: This subroutine transforms x and y coordinates of      *
* trajectory starting points to grid coordinates.                     *
*                                                                     *
***********************************************************************
*
      include 'includepar'
      include 'includecom'

      integer i,j
      logical error

      error=.false.

      if(numpoint.eq.0) goto 30
*
* TRANSFORM X- AND Y- COORDINATES OF STARTING POINTS TO GRID COORDINATES
*

      do i=1,numpoint
         xpoint(i)=(xpoint(i)-xlon0)/dx
         if (xglobal) then
           if (xpoint(i).gt.float(nx-1)) xpoint(i)=xpoint(i)-float(nx-1)
           if (xpoint(i).lt.0.) xpoint(i)=xpoint(i)+float(nx-1)
         endif
         ypoint(i)=(ypoint(i)-ylat0)/dy
      end do

15    continue
*
* CHECK IF STARTING POINTS ARE WITHIN DOMAIN
*
      do 25 i=1,numpoint

        if((xpoint(i).lt.0.).or.(xpoint(i).gt.float(nx-1)).or.
     &      (ypoint(i).lt.0.).or.(ypoint(i).gt.float(ny-1))) then

        write(*,*) ' NOTICE: STARTING POINT OUT OF DOMAIN HAS '//
     &              'BEEN DETECTED --> '
        write(*,*) ' IT IS REMOVED NOW ... '
        write(*,*) ' COMMENT: ',compoint(i)

          if(i.lt.numpoint) then

            do j=i+1,numpoint

              xpoint(j-1)=xpoint(j)
              ypoint(j-1)=ypoint(j)
              zpoint(j-1)=zpoint(j)
              kind(j-1)=kind(j)
              kindz(j-1)=kindz(j)
              compoint(j-1)=compoint(j)         
            end do
          endif

          numpoint=numpoint-1
          if(numpoint.gt.0) goto 15

        endif

25    continue

30    if(numpoint.eq.0) then

         error=.true.
         write(*,*) ' TRAJECTORY MODEL SUBROUTINE COORDTRAFO: '//
     &              'ERROR ! '
         write(*,*) ' NO TRAJECTORY STARTING POINTS ARE GIVEN !!!'

      endif

      return
      end 
