      PROGRAM  R A C F L
      IMPLICIT REAL*8  (A-H,O-Z)
C     CHARACTER*10  FDT*9, YMD,HMS,STT
      CHARACTER*9  FDT*9
      COMMON / RCLCM / IADRAB(120), RACL(500000)
!      COMMON / RCLCM / IADRAB(120), RACL(20000)

      MUL = 2
      WRITE(6,640)
C     
      READ(5,500)  MAXND, LTMAX, IFILE
C      Input the maximum number of d-bosons (MAXND), 
C      Maximum total angular momentum (LTMAX) and 
C      File # of the output file
C 
      OPEN(IFILE,FILE='racfl.dat',STATUS='UNKNOWN',ACCESS='SEQUENTIAL',
     &     FORM='UNFORMATTED')
C
      FDT='         '
C     If possible, the date of running the programs should be put in
C     "FDT" in nine characters.
C  In many systems,
C     CALL  DATE ( FDT )
C  In SUN in the education system in Kansai Univ. Takatsuki Campus.
C     CALL DATE_AND_TIME(YMD,HMS,STT,J)
C     FDT=YMD(7:8)//'.'//YMD(5:6)//'.'//YMD(3:4)
      MAXL = MAXND*2
      MDIM = MAXL
      IF( IFILE .EQ. 0 )  GOTO  10
C*******************************************************************
      REWIND  IFILE
10    WRITE(6,610)  FDT, MAXND, IFILE
C*******************************************************************
      ISQ = 0
      DO 230  IA =  2, MAXL
      DO 230  IB = IA, MAXL
        MADR = ( ( IA-2 ) * ( 2*MDIM-IA+1 ) ) / 2 + IB - IA + 1
        IADRAB( MADR ) = ISQ
        DO 240  IC = IA, MAXL
        DO 240  ID =  2, MAXL
          IF( IC .EQ. IA .AND. ID .LT. IB )  GOTO  240
          ISQ = ISQ + 1
240     CONTINUE
230   CONTINUE
      ISQDIM = ISQ
      IF( IFILE .GT. 0 )  THEN
        WRITE( IFILE )  MDIM, LTMAX, ISQDIM, FDT
        WRITE( IFILE )  ( IADRAB( I ), I = 1, ISQDIM )
      ENDIF
      WRITE(6,650)  MDIM, LTMAX,ISQDIM

      MUL2 = MUL*2
      DO 200  LT = 1, LTMAX
        ISQ = 0
        DO 210  IA =  2, MAXL
        DO 210  IB = IA, MAXL
          DO 220  IC = IA, MAXL
          DO 220  ID =  2, MAXL
            IF( IC .EQ. IA .AND. ID .LT. IB )  GOTO  220
            ISQ = ISQ + 1
            RACL( ISQ ) = DRAC( IA*2, IB*2, IC*2, ID*2, LT*2, MUL2 )
220       CONTINUE
210     CONTINUE
        IF( IFILE .EQ. 0 )  GOTO  20
C*******************************************************************
        WRITE(IFILE)  LT
        WRITE(IFILE) ( RACL( I ), I = 1, ISQ )
20      WRITE(6,620)  LT, MUL, MAXND, IFILE, ISQ
200   CONTINUE
      IF( IFILE .LE. 0 )  GOTO  30
      LT = -1
      WRITE(IFILE)  LT
C*******************************************************************
      ENDFILE IFILE
      REWIND   IFILE
30    STOP  00000
500   FORMAT(16I5)
610   FORMAT(//T7,'**  RACFL  EXECUTED   DATE : ',A8
     &   //T7,'RAC2  UPTO  ND=',I2,'  WRITTEN  ON  FILE-',I2/)
620   FORMAT(//T7,'RACL  FOR  L =',I2,'  MUL =',I2
     &   //T12,'UPTO  ND=',I2,'  WRITTEN  ON  FILE-',I2
     &   //T12,'# OF RAC =',I7 /)
630   FORMAT(//T7,'**  RCLFL  FINISHED    CPU-TIME=',I4,' (SEC)'//)
640   FORMAT(//T5,'PLEASE  INPUT',T22,'(1)  MAXIMUM # OF D-BOSONS',
     &   /T22,'(2)  MAXIMUM  L(TOTAL)' /T22,'(3)  FILE-#'
     &   //T10,'IN  I5 - FORMAT' / )
650   FORMAT(//T5,'**  RACFL  **'//T7,'MDIM =',I8,
     &         5X,'LTMAX =',I3,5X,'ISQDIM =',I9 /)
      END
      FUNCTION   D R A C  (JA1, JB1, LB1, LA1, JC1, LC1 )
      IMPLICIT REAL*8(A-H,O-Z)
!      COMMON / BCODCM / Q(52,52)
      COMMON / BCODCM / Q(104,104)
      DIMENSION  FUGO(2)
C*    DATA  IM  / ZFFFF0000 /
      DATA  FUGO / 1.D0, -1.D0 /
C   ** Q(I,M) (I.GE.M) BINOMIAL COEF. C(I-1,M-1)
C   ** Q(M,I) (I.GE.M) Q(I,M)**(-1/2)
C*    IE(K) = IAND(IM,K)
C** **
      JA = JA1
      JB = JB1
      LB = LB1
      LA = LA1
      JC = JC1
      LC = LC1
      GOTO  20
      ENTRY DRACAH(AJ,BJ,BL,AL,CJ,CL)
      S = 1
      IFIXJ = AJ + BJ + BL + AL + 0.1
      IF( IAND( IFIXJ, 1 ) .EQ. 1 )  S = - S
    7 JA=2*AJ+0.1
      JB=2*BJ+0.1
      JC=2*CJ+0.1
      LA=2*AL+0.1
      LB=2*BL+0.1
      LC=2*CL+0.1
      GO TO 1
C** ** ARGUMENT 2*J'S  INTEGER TYPE
   20 S = 1
      IF( IAND((JA+JB+LB+LA)/2, 1) .EQ. 1 )  S = - 1
!    1 IF(Q(13,3).NE.66.0D0)  CALL   B C O D I N  (52)
    1 IF(Q(13,3).NE.66.0D0)  CALL   B C O D I N  (104)
C     IF((IE(JA+JB-JC) + IE(JA+LB-LC) + IE(LA+LB-JC) + IE(LA+JB-LC)
C    (  + IE(JB+JC-JA) + IE(JB+LC-LA) + IE(LB+LC-JA) + IE(LB+JC-LA)
C    (  + IE(JC+JA-JB) + IE(JC+LA-LB) + IE(LC+LA-JB) + IE(LC+JA-LB))
C    3 .LT. 0 )  GOTO 70
      IF( (JA+JB-JC) .LT. 0 .OR. (JA+LB-LC) .LT. 0 .OR.
     &    (LA+LB-JC) .LT. 0 .OR. (LA+JB-LC) .LT. 0 .OR.
     &    (JB+JC-JA) .LT. 0 .OR. (JB+LC-LA) .LT. 0 .OR.
     &    (LB+LC-JA) .LT. 0 .OR. (LB+JC-LA) .LT. 0 .OR.
     &    (JC+JA-JB) .LT. 0 .OR. (JC+LA-LB) .LT. 0 .OR.
     &    (LC+LA-JB) .LT. 0 .OR. (LC+JA-LB) .LT. 0 )  GOTO  70
      MT=(JA+JB+JC)/2+1
      NA=MT-JA
      NB=MT-JB
      NC=MT-JC
      MA=(JA+LB+LC)/2+1
      KA=MA-LC
      MB=(LA+JB+LC)/2+1
      KB=MB-LC
      MC=(LA+LB+JC)/2+1
      KC=MC-JC
      KMIN=MAX0(MT,MA,MB,MC)+1
      KMAX=MIN0(MA+NA,MB+NB,MC+NC)
!      IF(KMAX.GT.52)GO TO 60
      IF(KMAX.GT.104)GO TO 60
      S=S*Q(JA+2,MA+1)*Q(KA,JA+1)*Q(LA+2,MB+1)*Q(KB,LA+1)*Q(LA+2,MC+1)*
     &  Q(KC,LA+1)/(Q(JA+2,MT+1)*Q(NC,JA+1)*DFLOAT(LA+1))
      X=0
      DO 10 K=KMIN,KMAX
      X=-X+Q(K,MT+1)*Q(NA,K-MA)*Q(NB,K-MB)*Q(NC,K-MC)
   10 CONTINUE
      DRAC  = X * S *FUGO(IAND(KMAX,1) + 1 )
      RETURN
   70 CONTINUE
      DRAC  = 0
      RETURN
   60 WRITE(6,602)  JA1, JB1, LB1, LA1, JC1, LC1
      GO TO 70
  602 FORMAT('DRAC/DRACAH   OUT OF RANGE  /  ',
     &       'W(', 3(I4,'/2,'), I4,'/2:', I4,'/2,', I4,'/2)')
      END
C *** SUBROUTINE BCODIN(NM)
      SUBROUTINE BCODIN(NM)
      IMPLICIT REAL*8(A-H,O-Z)
!      COMMON /BCODCM/Q(2704) !2704=52*52
      COMMON /BCODCM/Q(10816) ! 10816=104*104
C   ** MAIN ROUTINE DE HITSUYO NA COMMON   CODED BY M.OKA
C   **  COMMON /BCODCM/Q(NM,NM)
C   ** Q(M,I) (I.GE.M) BINOMIAL COEF. C(I-1,M-1)
C   ** Q(I,M) (I.GE.M) Q(M,I)**(-1/2)
      DATA  NMAX / 0 /
C %%%    ORIGINAL  FORM  IN  THE  KOHCHAN  LIBRARY .
C %%%      FITTED  TO  FORTRAN  77  ( JUNE  1982 )
C %%% NA(I,J)=I+(J-1)*NM
      NA(I,J) = J + (I-1)*NM
C** **
      IF(NM.LT.3) GO TO 24
      IF(NMAX.EQ.NM) RETURN
      IF(NMAX.NE.0) GO TO 23
      NMAX=NM
      Q(1) = 1
      Q(2) = 1
      Q(NM+1) = 1
      Q(NM+2) = 1
      DO 10 I=3,NM
      IQ = NA(I,1)
      IQ1 = NA(I,I)
      IQ2 = NA(1,I)
      Q( IQ )=1
      Q(IQ1)=1
      Q(IQ2)=1
      DO 10 J=2,I-1
      IQ = NA(J,I)
      IQ1 = NA(J-1,I-1)
      IQ2 = NA(J,I-1)
      Q(IQ) = Q( IQ1 ) + Q( IQ2 )
   10 CONTINUE
      DO 20 J=3,NM
      DO 20 I=1,J-1
      IQ = NA(J,I)
      IQ1 = NA(I,J)
      Q( IQ ) = 1 / DSQRT( Q( IQ1 ) )
   20 CONTINUE
C     WRITE(6,600)  NM
  600 FORMAT(/T12,'***  BCODIN  CALLED  (NM=',I2,')  ***'/)
      RETURN
   23 WRITE(6,101)  NMAX,NM
      GO TO 25
   24 WRITE(6,100)  NM
   25 RETURN
  100 FORMAT(T15,'NMAX.LT.3 IN (BCODIN)  NMAX=',I4)
  101 FORMAT(T15,'/BCODCM/ LENGTH MISMATCHED  OLD NMAX=',I4,
     &       ', NEW NMAX =',I4)
      END
