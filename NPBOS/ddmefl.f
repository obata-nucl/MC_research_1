      PROGRAM  D D M E F L
      IMPLICIT  REAL*8  (A-H, O-Z)
C*****************************************************************************
C        FILE - 12  :  OUTPUT  ( D(+)D  MATRIX  ELEMENTS )
C*****************************************************************************
      PARAMETER   (IDDM=30000)
      COMMON / SPSTCM /  ISTBS(300),NBD(300),LD(300),NSEN(300),NDL(300)
     1                  ,NBBAS(16),ISTBS2(300),LAD(300),LADBS(16)
      COMMON / DDMECM /  IBSDD(300,2:4),ILDDM(IDDM),DDMAT(IDDM)
      DIMENSION          IRDDM(IDDM)
      COMMON / SR /  SR0,SR(500)
C
      OPEN(6,FILE='out3.dat',STATUS='UNKNOWN',ACCESS='SEQUENTIAL',
     &     FORM='FORMATTED')
C     OPEN(11,FILE='cfpdk.dat',STATUS='OLD',ACCESS='SEQUENTIAL',
C    &     FORM='UNFORMATTED')
      OPEN(11,FILE='cfp2.dat',STATUS='OLD',ACCESS='SEQUENTIAL',
     &     FORM='UNFORMATTED')
C     inputting the boson 2-body cfp
      OPEN(12,FILE='ddmefl.dat',STATUS='UNKNOWN',ACCESS='SEQUENTIAL',
     &     FORM='UNFORMATTED')
C     output file of matrix elements of d-boson 1-body operator
C
C     write(6,*) ' Input the maximum # of d-bosons in I5 format.'
      READ(5,500)  NDMAX
      NDMAX1 = NDMAX + 1
      CALL  COMIN1( NDMAX )
      ICFPPR = ICFPPR + 1
      DO 200  I = 1, 500
      SR( I ) = DSQRT( DFLOAT( I ) )
  200 CONTINUE
      SR0 = 0
      IBS = 0
      NST = NBBAS( NDMAX1 )
      NST1 = NST + 1
      DO 230  K = 2, 4
      IBSDD( 1, K ) = IBS
      DO 210  I = 1, NST
      IF( NBD(I) .LE. 0 )  GOTO  10
      DO 220  J = 1, I
      IF( NBD(J) .LT. NBD(I) )  GOTO 220
      IF( NBD(J) .GT.  NBD(I) )  GOTO  10
      IF( IABS(LD(J)-LD(I)) .GT. K .OR. (LD(J)+LD(I)) .LT. K )
     &    GOTO  220
      IBS = IBS + 1
      DDMAT( IBS ) = DDME( J, I, K, 0.D0, 0 )
      ILDDM( IBS ) = J
      IRDDM( IBS ) = I
  220 CONTINUE
   10 IBSDD( I+1, K ) = IBS
  210 CONTINUE
      IBSDD( NST+1, K ) = IBS
  230 CONTINUE
      NMEDD = IBS
C***********************************************************************
      WRITE(6,600)  NDMAX, NST, NMEDD
      WRITE( 12 )  NDMAX, NST, NMEDD
      DO 240  K = 2, 4
      WRITE(6,602)  K
      WRITE(6,610)  (I,IBSDD(I,K), I=1,NST1)
      WRITE( 12 )  (IBSDD(I,K), I=1,NST1)
C ***
      J1 = IBSDD(1,K) + 1
      J2 = IBSDD(NST+1,K)
      WRITE(6,604)  K, J1, J2
      WRITE(6,620)  (IRDDM(J),ILDDM(J),DDMAT(J),J=J1,J2)
      WRITE( 12 )  (ILDDM(J),DDMAT(J),J=J1,J2)
  240 CONTINUE
      ENDFILE 12
      STOP  00000
  500 FORMAT(2I5)
  600 FORMAT(//T6,'**  DDMEFL  EXECUTION  **'
     &   //T10,'NDMAX(INPUT) =',I6,5X,'NST =',I6,5X,'NMEDD =',I8)
  602 FORMAT(//T6,'**  K =',I2,'  BASE  ADDRESS  FOR  EACH  STATE')
  604 FORMAT(//T6,'**  K =',I2,'  (KET, BRA)  DD-ME      ME# =',I7,
     &   ' -->',I7 )
  610 FORMAT(/ (T9,10 ( '  (', I3, ')', I5 )) )
  620 FORMAT(/ (T9, 6 ( '  (',I4,'/',I3,')',F7.3)) )
      E N D

      FUNCTION   DDME ( JL, JR, K, C, IW )
      IMPLICIT  REAL*8  (A-H,O-Z )
      PARAMETER   (ICFP1=2907)
      COMMON / SPSTCM / ISTBS(300),NBD(300),LD(300),NSEN(300),NDL(300)
     &                 ,NBBAS(16),ISTBS2(300),LAD(300),LADBS(16)
      COMMON / CFP1CM / IDAU(ICFP1),L1(ICFP1),CFP1(ICFP1)
      COMMON / SR / SR0,SR(500)
C /--------------------------------------------------------------------/
C / ***   ( JL  !!  ( (D+) * (D-) ) (K)  !!  JR )  *  C            *** /
C /--------------------------------------------------------------------/
      DDME = 0
      IF( NBD( JL ) .NE. NBD( JR ) )  RETURN
      LL = LD( JL )
      LL2 = LL * 2
      LR = LD( JR )
      LR2 = LR * 2
      DBD = NBD( JL )
      IA = ISTBS( JR ) + 1
      IR = ISTBS( JR + 1 )
      MA = ISTBS( JL ) + 1
      MR = ISTBS( JL + 1 )
    4 IB = IDAU ( IA )
    5 MB = IDAU ( MA )
    6 IF( IB - MB )  7, 3, 9
    9 MA = MA + 1
      IF( MA .LE. MR ) GOTO  5
      GOTO 12
    7 IA = IA + 1
      IF( IA .GT. IR )  GOTO 12
      IB = IDAU ( IA )
      GOTO  6
    3 LX = LD( IB )
      LX2 = LX * 2
      X = CFP1( IA ) * CFP1( MA )
     +  * DRAC( LR2, 4, LL2, 4, LX2, K*2 )
      IF( IAND(1,LX) .EQ. 1 )  X = - X
      DDME = DDME + X
      IA = IA + 1
      MA = MA + 1
      IF( IA .LE. IR .AND. MA .LE. MR )  GOTO  4
   12 DDME = DDME * DBD * SR( K*2 + 1 ) * SR( LL2+1 ) * SR( LR2+1 )
      IF( IAND(1, K+LL) .EQ. 1 )  DDME = - DDME
      IF( C .NE. 0.D0 )  DDME = DDME * C
      IF( IW .EQ. 0 )  RETURN
      WRITE(6,600) JL, NBD(JL), LD(JL),  JR, NBD(JR), LD(JR), K, DDME
  600 FORMAT(T5,'FIN ST :',I3,'   (ND:',I2,'  LD:',I2,')  ;  INI ST :',
     &  I3,'   (ND:',I2,'  LD:',I2,')  ;  K :',I2,'   ** ME :',F8.4 )
      RETURN
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
!      COMMON /BCODCM/Q(2704)
      COMMON /BCODCM/Q(10816)
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

      SUBROUTINE  C O M I N 1  ( MAXND )
      IMPLICIT  REAL*8(A-H,O-Z)
      PARAMETER   (ICFP1=2907, ICFP2=6371)
      COMMON / SPSTCM /  ISTBS(300),NBD(300),LD(300),NSEN(300),NDL(300)
     1,NBBAS(16),ISTBS2(300),LAD(300),LADBS(16)
      COMMON / CFP1CM / IDAU(ICFP1),L1(ICFP1),CFP1(ICFP1)
      COMMON / CFP2CM / IDAU2(ICFP2),L2(ICFP2),CFP2(ICFP2),LIM(ICFP2)
      COMMON / OUTCM / ICMW, IDBG, NPSTW, MATW, IWCF
C     REWIND  11
C ****
      READ( 11,END=100 )  NBX, ( NBBAS(J), J = 1, NBX )
      IR = NBX-1
      WRITE(6,600)  MAXND, IR
C ****
      IF( MAXND .GT. IR )  MAXND = IR
C ****
      NBX = MAXND + 1
      IR = NBBAS( NBX )
      READ( 11,END=100 )  ( NBD   (J), J = 1, IR )
      READ( 11,END=100 )  ( LD    (J), J = 1, IR )
      READ( 11,END=100 )  ( NSEN  (J), J = 1, IR )
      READ( 11,END=100 )  ( NDL   (J), J = 1, IR )
      DO 201  J = 1, IR
  201 LD(J) = IABS( LD(J) )
      IR1 = IR + 1
      READ( 11,END=100 )  ( ISTBS(J) , J = 1, IR1 )
      READ( 11,END=100 )  ( ISTBS2(J), J = 1, IR1 )
      IL = ISTBS( IR + 1 )
      READ( 11,END=100 )  ( IDAU (J), J = 1, IL )
      READ( 11,END=100 )  ( L1   (J), J = 1, IL )
      READ( 11,END=100 )  ( CFP1 (J), J = 1, IL )
      DO 202  J = 1, IL
  202 L1(J) = LD(IDAU(J))
      IL = ISTBS2( IR + 1 )
      READ( 11,END=100 )  ( IDAU2(J), J = 1, IL )
      READ( 11,END=100 )  ( LIM  (J), J = 1, IL )
      READ( 11,END=100 )  ( L2   (J), J = 1, IL )
      READ( 11,END=100 )  ( CFP2 (J), J = 1, IL )
      DO 203  J = 1, IL
  203 LIM(J) = IABS( LIM(J) )
      DO 204  J = 1, IL
  204 L2(J) = IABS( L2(J) )
      RETURN
  100 WRITE(6,680)
      ICMW = 3
      RETURN
  600 FORMAT(/T12,'# CFP FILE (#11) READ-IN   REQUIRED MAX(ND) =',
     &   I2,'  / MAX(ND) ON FILE =',I2,' (IF FORMER .GT. LATTER,',
     &   ' LATTER BE TAKEN)'/)
  680 FORMAT(//T7,'----->   ABNORMAL  TERMINATION  IN  READING',
     +  '  "CFPDK" .   ICMW  WAS  SET  3 .'//)
      END
