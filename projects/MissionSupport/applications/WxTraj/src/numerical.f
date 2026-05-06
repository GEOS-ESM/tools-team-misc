C THESE ROUTINES HAVE BEEN TAKEN FROM PRESS ET AL. (1992): NUMERICAL RECIPES
C HOWEVER, THEY HAVE BEEN MODIFIED FOR PERFORMANCE REASONS.
C A. Stohl, 31 May 1994


      SUBROUTINE BICUBIC(Y,Y1,Y2,Y12,X1L,X1U,X2L,X2U,X1,X2,ANSY,LD1,LD2)
      DIMENSION Y(4,LD1,LD2),Y1(4,LD1,LD2),Y2(4,LD1,LD2),Y12(4,LD1,LD2)
      DIMENSION C(4,4,2,3),ANSY(LD1,LD2)
      CALL BCUCOF(Y,Y1,Y2,Y12,X1U-X1L,X2U-X2L,C,LD1,LD2)
      T=(X1-X1L)/(X1U-X1L)
      U=(X2-X2L)/(X2U-X2L)
      DO 11 M=1,LD1
        DO 11 N=1,LD2
          ANSY(M,N)=0.
          DO 11 I=4,1,-1
11          ANSY(M,N)=T*ANSY(M,N)+((C(I,4,M,N)*U+C(I,3,M,N))*U+
     +      C(I,2,M,N))*U+C(I,1,M,N)
      RETURN
      END
 

      SUBROUTINE BCUCOF(Y,Y1,Y2,Y12,D1,D2,C,LD1,LD2)
      DIMENSION C(4,4,2,3),Y(4,LD1,LD2),Y1(4,LD1,LD2)
      DIMENSION Y2(4,LD1,LD2),Y12(4,LD1,LD2)
      DIMENSION CL(16),X(16),WT(16,16)
      SAVE WT
      DATA WT/1.,0.,-3.,2.,4*0.,-3.,0.,9.,-6.,2.,0.,-6.,
     *  4.,8*0.,3.,0.,-9.,6.,-2.,0.,6.,-4.,10*0.,9.,-6.,
     *  2*0.,-6.,4.,2*0.,3.,-2.,6*0.,-9.,6.,2*0.,6.,-4.,
     *  4*0.,1.,0.,-3.,2.,-2.,0.,6.,-4.,1.,0.,-3.,2.,8*0.,
     *  -1.,0.,3.,-2.,1.,0.,-3.,2.,10*0.,-3.,2.,2*0.,3.,
     *  -2.,6*0.,3.,-2.,2*0.,-6.,4.,2*0.,3.,-2.,0.,1.,-2.,
     *  1.,5*0.,-3.,6.,-3.,0.,2.,-4.,2.,9*0.,3.,-6.,3.,0.,
     *  -2.,4.,-2.,10*0.,-3.,3.,2*0.,2.,-2.,2*0.,-1.,1.,
     *  6*0.,3.,-3.,2*0.,-2.,2.,5*0.,1.,-2.,1.,0.,-2.,4.,
     *  -2.,0.,1.,-2.,1.,9*0.,-1.,2.,-1.,0.,1.,-2.,1.,10*0.,
     *  1.,-1.,2*0.,-1.,1.,6*0.,-1.,1.,2*0.,2.,-2.,2*0.,-1.,1./
      D1D2=D1*D2
      DO 15 M=1,LD1
        DO 15 N=1,LD2
          DO 11 I=1,4
            X(I)=Y(I,M,N)
            X(I+4)=Y1(I,M,N)*D1
            X(I+8)=Y2(I,M,N)*D2
11          X(I+12)=Y12(I,M,N)*D1D2
          DO 13 I=1,16
            XX=0.
            DO 12 K=1,16
12            XX=XX+WT(I,K)*X(K)
13          CL(I)=XX
          L=0
          DO 15 I=1,4
            DO 15 J=1,4
              L=L+1
15            C(I,J,M,N)=CL(L)
      RETURN
      END




      SUBROUTINE POLYNOM(XA,YA,N,X,Y,LD)
      PARAMETER (NMAX=10)
      DIMENSION XA(N),YA(LD,N),C(2,NMAX),D(2,NMAX),Y(LD)
      NSS=1
      DIF=ABS(X-XA(1))
      DO 11 I=1,N
        DIFT=ABS(X-XA(I))
        IF (DIFT.LT.DIF) THEN
          NSS=I
          DIF=DIFT
        ENDIF
        DO 11 MM=1,LD
        C(MM,I)=YA(MM,I)
11      D(MM,I)=YA(MM,I)

      DO 13 MM=1,LD
        NS=NSS
        Y(MM)=YA(MM,NS)
        NS=NS-1
      DO 13 M=1,N-1
        DO 12 I=1,N-M
          HO=XA(I)-X
          HP=XA(I+M)-X
          W=C(MM,I+1)-D(MM,I)
          DEN=HO-HP
          DEN=W/DEN
          D(MM,I)=HP*DEN
          C(MM,I)=HO*DEN
12      CONTINUE
        IF (2*NS.LT.N-M)THEN
          DY=C(MM,NS+1)
        ELSE
          DY=D(MM,NS)
          NS=NS-1
        ENDIF
        Y(MM)=Y(MM)+DY
13    CONTINUE
      RETURN
      END
