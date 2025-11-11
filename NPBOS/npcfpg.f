C
C  program NPCFPG generates 2-body d-boson cfp from 1-body d-boson cfp.
C
C  input data example
C     ....*....1
C        11             ... NDX (I5)
C
C*DECK NPCFPG
      PROGRAM  NPCFPG
C     (input,output,tape5=input,tape6=output,tape7,
C    ?              tape2=output,
C    1                tape3,tape11)
      IMPLICIT REAL*8(A-H,O-Z)
C     LEVEL 2, IDAU,LD1,CFP1,IDAU2,L2,CFP2,LIM
      COMMON / REDMAT / REDGS(546,5), BETFAC(72,2), IBDEL(2,6)
      COMMON/SPSTCM/ISTBS(960),NBD(960),LD(960),NSEN(960),NDEL(960),
     ?             NBBAS(20),ISTBS2(960)
      COMMON/CFP1CM/  IDAU(20000),LD1(20000),CFP1(20000)
      COMMON/CFP2CM/IDAU2(48000),L2(48000),CFP2(48000),LIM(48000)
      COMMON/BCODCM/Q(80,80)
      COMMON/RACCM/G(121)
      DATA  NDM, IDIB, IDB, IDGS / 16, 6, 72, 546 /
C

      OPEN(2,FILE='out2.dat',STATUS='UNKNOWN',FORM='FORMATTED',
     &     ACCESS='SEQUENTIAL')
      OPEN(3,FILE='cfp1.dat',STATUS='OLD',FORM='UNFORMATTED',
     &     ACCESS='SEQUENTIAL')
      OPEN(6,FILE='out3.dat',STATUS='UNKNOWN',FORM='FORMATTED',
     &     ACCESS='SEQUENTIAL')
      OPEN(7,STATUS='SCRATCH',FORM='UNFORMATTED',ACCESS='SEQUENTIAL')
      OPEN(11,FILE='cfp2.dat',STATUS='UNKNOWN',FORM='UNFORMATTED',
     &     ACCESS='SEQUENTIAL')
C
      CALL  REDRED(NDM,IDGS,IDB,IDIB,REDGS,BETFAC,IBDEL)
C
C----------- READ THE MAXIMAL NUMBER FOR WHITCH THE CFP S HAVE TO BE
C                    CONSTRUCTED
      write(6,*) ' Input the maximum # of d-bosons in I5 format.'
      read(5,501)  NDX
C
C     CALL RACSU
      NBD(1) = 0
      LD(1) = 0
      NSEN(1) = 0
      NDEL(1) = 0
      NBBAS(1) = 0
      M = 1
      DO 201  ND = 1, NDX
      LX = 2*ND + 1
      DO 202  L1 = 1, LX
      L = L1 - 1
      NBX = ND/2 + 1
      DO 203  NBR = 1, NBX
      NB = NBX - NBR
      NDLX = ND/3 + 1
      DO 204  NDL1 = 1, NDLX
      NDL = NDL1 - 1
      MM = ND - 2*NB - 3*NDL
      IF( MM .LT. 0 )  GOTO  204
      IF( MM .GT. L )  GOTO  204
      IF( 2*MM .LT. L )  GOTO  204
      IF( L .EQ. 2*MM-1 )  GOTO 204
      M = M + 1
      NBD( M ) = ND
      LD( M ) = L
      NSEN( M ) = ND - 2*NB
      NDEL( M ) = NDL
  204 CONTINUE
  203 CONTINUE
  202 CONTINUE
  201 CONTINUE
      NST = M
      WRITE(6,601)  NDX, NST
      ND = 0
      DO 205  I = 1, NST
      IF( NBD(I) .EQ. ND )  GOTO  205
      NBBAS(ND+1) = I-1
      ND = NBD(I)
  205 CONTINUE
      NBBAS(ND+1) = NST
      NDX1 = NDX + 1
      WRITE(6,611)  ( NBBAS(I), I = 1, NDX1 )
      J = 0
      DO 212  I = 2, NST
      ND = NBD(I)
      KL = NBBAS(ND-1)+1
      KR = NBBAS(ND)
      L = LD(I)
      DO 211  K = KL, KR
      IF( IABS( LD(K)-L ) .GT. 2 )  GOTO  211
      IF( LD(K)+L .LT. 2 )  GOTO  211
      J = J + 1
      IDAU(J) = K
      LD1(J) = LD(K)
      NBDF = (1+NSEN(K)-NSEN(I))/2
      NB = ( ND - 1 - NSEN(K) ) / 2
      NCDF = NDEL(I) - NDEL(K)
      LX = LD(K) - L + 3
C*****
      X = RED(ND-1,NB,NDEL(K),LD(K),LX,NBDF,NCDF)
C*****
C     IF( ( (L+LD(K)) .AND. 1 ) .EQ. 1 )  X = - X
      LLDK=L+LD(K)
      LLDKA1=IAND(LLDK,1)
      IF(LLDKA1.EQ.1) X=-X
      Y = SQRT( DFLOAT( ND*(2*L+1) ) )
      CFP1(J) = X / Y
  211 CONTINUE
      ISTBS(I+1) = J
  212 CONTINUE
      ISTBS(NST+1) = J
      NDX1 = NDX+1
      NST1 = NST + 1
      WRITE(7)  NDX1, ( NBBAS(I), I = 1, NDX1 )
      WRITE(7)  ( NBD(I), I = 1, NST )
      WRITE(7)  ( LD(I), I = 1, NST )
      WRITE(7)  ( NSEN(I), I = 1, NST )
      WRITE(7)  ( NDEL(I), I = 1, NST )
      WRITE(7)  ( ISTBS(I), I = 1, NST1 )
      WRITE(6,602)( ISTBS(I),I=1,NST1)
  602 FORMAT(1X,10I10)
      M = ISTBS( NST1 )
      WRITE(7)  ( IDAU(I), I = 1, M )
      WRITE(7)  ( LD1(I), I = 1, M )
      WRITE(7)  ( CFP1(I), I = 1, M )
      WRITE(6,605) (CFP1(I),I=1,M)
  605 FORMAT(1X,15F8.5)
      REWIND 7
      CALL CFP2NP(NDX)
      REWIND  7
  501 FORMAT(16I5)
  601 FORMAT(/// T7,'***  # OF D-BOSON =',I3,T40,'# OF STATES =',
     1   I5  //  )
  611 FORMAT( ///T7,'#NBBAS ----'// T12,12I8)
      STOP
      END
C*DECK CFP2NP
C
      subroutine CFP2NP(NBODY)
C
      IMPLICIT REAL*8(A-H,O-Z)
C     LEVEL 2, IDAU,L1,CFP1,IDAU2,L2,CFP2,LIM
      COMMON/SPSTCM/ISTBS(960),NBD(960),LD(960),NSEN(960),NDL(960),
     ?            NBBAS(20),ISTBS2(960)
      COMMON/CFP1CM/IDAU (20000),L1(20000),CFP1(20000)
      COMMON/CFP2CM/IDAU2(48000),L2(48000),CFP2(48000),LIM(48000)
      DIMENSION  ND(1)
      EQUIVALENCE  (ND(1),NDL(1))
      NBODY = IABS( NBODY )
      NBODY1 =  NBODY + 1
      READ( 7 )  NBX, ( NBBAS(J), J = 1, NBX )
      JE = NBBAS( NBX )
      READ( 7 )  ( NBD  (J), J = 1, JE )
      READ( 7 )  ( LD   (J), J = 1, JE )
      READ( 7 )  ( NSEN (J), J = 1, JE )
      READ( 7 )  ( NDL  (J), J = 1, JE )
      JE1 = JE + 1
      READ( 7 )  ( ISTBS(J), J = 1, JE1 )
      JE1 = ISTBS(JE+1)
      READ( 7 ) ( IDAU(J), J = 1, JE1 )
      READ( 7 ) ( L1  (J), J = 1, JE1 )
      READ( 7 ) ( CFP1(J), J = 1, JE1 )
      NCFP = 3
      ISTBS2(1)=0
      ISTBS2(2)=0
      DO 201  J = 1, 3
      CFP2(J) = 1.D0
      LIM(J) = (J-1) * 2
      ISTBS2(J+2) = J - 1
      IDAU2(J) = 1
  201 CONTINUE
      NBSE = NBBAS(NBODY+1)
      DO 202  J1 = 6, NBSE
      ISTBS2(J1) = NCFP
      NX = NBD(J1)
      LX = LD(J1)
      IL1 = ISTBS(J1) + 1
      IR1 = ISTBS(J1+1)
      DO 203  LLZ = 1, 3
      LJ2 = (LLZ-1) * 2
      FLJ2=LJ2
      LDL = IABS( LX-LJ2 )
      LDR = LX + LJ2
      LDP = LDL
      NX2 = NX - 2
      NBSL = NBBAS(NX2) + 1
      NBSR = NBBAS(NX2+1)
      DO 204  JX = NBSL, NBSR
    1 IF( LD(JX) - LDP )  204, 3, 4
    4 LDP = LDP + 1
      IF( LDP .LE. LDR )  GO  TO  1
      GO  TO  203
    3 NCFP = NCFP + 1
      L2(NCFP) = LDP
      FLDP = LDP
      IDAU2(NCFP) = JX
      LIM(NCFP) = LJ2
      CFP2X = 0.D0
      DO 205  KX = IL1, IR1
      NKX = IDAU(KX)
      LKX = LD(NKX)
      IF( IABS( LDP-LKX ) .GT. 2 )  GO  TO  205
      NKL = ISTBS(NKX) + 1
      NKR = ISTBS(NKX + 1)
      DO 206  KD = NKL, NKR
      IF( IDAU(KD) .EQ. JX )  GO  TO  2
  206 CONTINUE
      GO  TO  205
    2 FLKX = LKX
      CFP2X = CFP2X + CFP1(KX) * CFP1(KD) * SQRT( (FLKX+FLKX+1.D0) *
     1  (FLJ2+FLJ2+1.D0) ) * DRAC  (2*LDP,4,2*LX,4,2*LKX,2*LJ2)
  205 CONTINUE
C***
      CFP2(NCFP) = CFP2X
C***
  204 CONTINUE
  203 CONTINUE
  202 CONTINUE
      ISTBS2(NBSE+1) = NCFP
C+++++++
C+++++     TAKEN  FROM  PROGRAM  PRDCFP  IN  PART
C+++++++
      I = NBBAS(NBODY+1)
      M = ISTBS(I+1)
      I1=I+1
      WRITE(6,100) (ISTBS2(J),J=1,I1)
  100 FORMAT(1X,10I10/)
      WRITE(11) NBODY1,(NBBAS(J),J=1,NBODY1)
      WRITE(11) (NBD(J),J=1,I)
      WRITE(11) (LD (J),J=1,I)
      WRITE(11) (NSEN(J),J=1,I)
      WRITE(11) (NDL (J),J=1,I)
      WRITE( 11 )  ( ISTBS(J), J = 1, I1 )
      WRITE( 11 )  ( ISTBS2(J), J = 1, I1 )
      WRITE( 11 )  ( IDAU (J), J = 1, M )
      WRITE( 11 )  ( L1   (J), J = 1, M )
      WRITE( 11 )  ( CFP1 (J), J = 1, M )
      M = ISTBS2(I+1)
      WRITE( 11 )  ( IDAU2(J), J = 1, M )
      WRITE( 11 )  ( LIM  (J), J = 1, M )
      WRITE( 11 )  ( L2   (J), J = 1, M )
      WRITE( 11 )  ( CFP2 (J), J = 1, M )
  101 FORMAT(1X,15F8.5)
      WRITE(6,101) (CFP2(J),J=1,M)
C=====
      WRITE(6,600)
C+++
C+++
  501 FORMAT(10I5)
  600 FORMAT( '1' // )
      STOP  '  NORMAL  TERMINATION  '
      END
C  'REDRED' WAS ADDED IN TOKYO
C*DECK REDRED
      SUBROUTINE REDRED(NDM,IDGS,IDB,IDIB,REDGS,BETFAC,IBDEL)
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION REDGS(IDGS,5),BETFAC(IDB,2),IBDEL(2,IDIB)
      COMMON/RACCM/G(121)
      READ(3) NDM,IDGS,IDB,IDIB
      READ(3) REDGS,BETFAC,IBDEL
      READ(3) G
      RETURN
      END
C*DECK RED
      FUNCTION RED(ND,NB,NC,L,LD,NBD,NCD)
C
C      RED=<ND+1,NB+NBD,NC+NCD,LPR//D+//ND,NB,NC,L>  ; LD=L-LPR+3
C      LD NEED TO LIE BETWEEN 1 AND 5  NO CHECK IS MADE ON THIS
C      VALUE RETURNED ONLY NON ZERO IF :
C          NBD=0 OR 1 AND NBD+NCD=0 OR 1
C       AND K,KP<=L,LP<=2*K,2*KP  ; K=ND-2*NB-3*NC
C
      IMPLICIT REAL*8(A-H,O-Z)
      COMMON/REDMAT/REDGS(546,5),BETFAC(72,2),IBDEL(2,6)
      RED=0.D0
      LPR=L+3-LD
      IF(ND.LT.0) GOTO 999
      IF(NB.LT.0) GOTO 999
      IF(NBD) 999,100,101
  101 IF(NBD-1) 999,200,999
C
  100 CONTINUE
C      DIFFERENCE IN NBETA(=(ND-SENIORITY)/2) EQUAL TO 0
C
      IF(NC.LT.0) GOTO 999
      IF(NCD) 999,110,111
  111 IF(NCD-1) 999,120,999
C
  110 CONTINUE
C      CND=0
C
      ND3=ND-2*NB-3*NC
      IF(ND3) 999,112,112
  112 IF(L.LT.ND3) GOTO 999
      IF(L.GT.2*ND3) GOTO 999
      K=ND/2
      RED=BETFAC(NB+1+(K+1)*(ND-K),1)
      K=ND3*(ND3-1)/2+1+L+IBDEL(1,NC+1)
      RED=RED*REDGS(K,LD)
      RETURN
C
  120 CONTINUE
C      NCD=1
C      NC = MIN
C
      ND3=ND-2*NB-3*NC-2
      IF(ND3) 999,122,122
  122 IF(LPR.LT.ND3) GOTO 999
      IF(LPR.GT.2*ND3) GOTO 999
      K=ND/2
      RED=BETFAC(NB+1+(K+1)*(ND-K),1)
      NCP=NC+1
      K=ND3*(ND3-1)/2+1+LPR+IBDEL(2,NCP)
      RED=RED*REDGS(K,LD)
      RETURN
C
  200 CONTINUE
C      NBD=1
C
      IF(NCD) 211,210,999
  211 IF(NCD+1) 999,220,999
C
  210 CONTINUE
C      NCD=0
C
      IF(NC.LT.0) GOTO 999
      ND3=ND-2*NB-3*NC-1
      IF(ND3) 999,212,212
  212 IF(LPR.LT.ND3) GOTO 999
      IF(LPR.GT.2*ND3) GOTO 999
      K=ND/2
      RED=BETFAC(NB+1+(K+1)*(ND-K),2)*(-1)**(L+LPR)
      K=ND3*(ND3-1)/2+1+LPR+IBDEL(1,NC+1)
      RED=RED*REDGS(K,6-LD)
      RETURN
C
  220 CONTINUE
C      CND=-1
C      NC = MAX
C
      IF(NC.LT.1) GOTO 999
      ND3=ND-2*NB-3*NC
      IF(ND3) 999,222,222
  222 IF(L.LT.ND3) GOTO 999
      IF(L.GT.2*ND3) GOTO 999
      NCM=NC-1
      IF(NCM) 999,221,221
  221 CONTINUE
      K=ND/2
      RED=BETFAC(NB+1+(K+1)*(ND-K),2)*(-1)**(L+LPR)
      K=ND3*(ND3-1)/2+1+L+IBDEL(2,NC)
      RED=RED*REDGS(K,6-LD)
      RETURN
C
  999 RED=0.D0
      RETURN
      END
C*DECK DSIXJ
c      REAL FUNCTION DSIXJ*8(AJ,BJ,CJ,AL,BL,CL)
      REAL FUNCTION DSIXJ(AJ,BJ,CJ,AL,BL,CL)
      IMPLICIT REAL*8(A-H,O-Z)
C** ** RAPID RACAH/SIXJ PROGRAM BY N.HOSHI (FEB.1976) VERSION 002.03
C** **  BASED ON 'SIXJ' BY K.SHIMIZU WITH THE IDEAS OF H.NISHIMURA
C** ** REVISED BY M. OKA FOR THE VOS3 SYSTEM (AUGUST 1980)
      REAL*4 AJ,BJ,CJ,AL,BL,CL,ADUMMY
      DIMENSION NERR(20)
C** ** ARGUMENT J'S  REAL*4 TYPE
      COMMON /BCODCM/QQ(80,80)
      Q(I,M)=QQ(M,I)
C   ** Q(I,M) (I.GE.M) BINOMIAL COEF. C(I-1,M-1)
C   ** Q(M,I) (I.GE.M) Q(I,M)**(-1/2)
C     REAL*8 FUGO(2)/1.0D0,-1.0D0/;FG(K)=FUGO((K.AND.'00000001'Z)+1)
      FG(K)=FUGO(K)
      ASS(ADUMMY)=SIGN(0.1,ADUMMY)
C     INTEGER NERR(20)
C     DATA NERR/'ARG.',' RAN','GE O','VER ','IN ('/
C     DATA NNNN/Z80000001/
      DATA NNNN/-2147483647/
C** **
      S=1.0D0
      IND=1
      GO TO 7
      ENTRY DRACAH(AJ,BJ,BL,AL,CJ,CL)
      S=FG(IFIX(AJ+BJ+BL+AL+0.1))
      IND=2
      GOTO  7
      ENTRY DRAC(JA,JB,LB,LA,JC,LC)
C** ** ARGUMENT 2*J'S  INTEGER TYPE
      S=FG((JA+JB+LB+LA)/2)
      IND=2
      GO TO 1
      ENTRY D6J(JA,JB,JC,LA,LB,LC)
      S=1.0D0
      IND=1
      GOTO  1
    7 JA=2.0*AJ+ASS(AJ)
      JB=2.0*BJ+ASS(BJ)
      JC=2.0*CJ+ASS(CJ)
      LA=2.0*AL+ASS(AL)
      LB=2.0*BL+ASS(BL)
      LC=2.0*CL+ASS(CL)
    1 IF(Q(13,3).NE.66.0D0)CALL BCODIN(80)
C     IEE=((JA+JB-JC).OR.(JA+LB-LC).OR.(LA+LB-JC).OR.(LA+JB-LC)
C    +.OR.(JB+JC-JA).OR.(JB+LC-LA).OR.(LB+LC-JA).OR.(LB+JC-LA)
C    +.OR.(JC+JA-JB).OR.(JC+LA-LB).OR.(LC+LA-JB).OR.(LC+JA-LB))
C    +.AND.'80000001'Z
      IEE=IOR(JA+JB-JC,JA+LB-LC)
      IEE=IOR(IEE,LA+LB-JC)
      IEE=IOR(IEE,LA+JB-LC)
      IEE=IOR(IEE,JB+JC-JA)
      IEE=IOR(IEE,JB+LC-LA)
      IEE=IOR(IEE,LB+LC-JA)
      IEE=IOR(IEE,LB+JC-LA)
      IEE=IOR(IEE,JC+JA-JB)
      IEE=IOR(IEE,JC+LA-LB)
      IEE=IOR(IEE,LC+LA-JB)
      IEE=IOR(IEE,LC+JA-LB)
      IEE=IAND(IEE,NNNN)
      IF(IEE.NE.0) GO TO 70
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
      IF(KMAX.GT.80)GO TO 60
      S=S*Q(JA+2,MA+1)*Q(KA,JA+1)*Q(LA+2,MB+1)*Q(KB,LA+1)*Q(LA+2,MC+1)*
     +Q(KC,LA+1)/(Q(JA+2,MT+1)*Q(NC,JA+1)*DFLOAT(LA+1))
      X=0.D0
      DO 10 K=KMIN,KMAX
      X=-X+Q(K,MT+1)*Q(NA,K-MA)*Q(NB,K-MB)*Q(NC,K-MC)
   10 CONTINUE
      D6J=X*S*FG(KMAX)
      RETURN
   70 D6J=0.D0
      RETURN
   60 IF(IND.NE.1)GO TO 66
C     ENCODE(56,601,NERR(6))JA,JB,JC,LA,LB,LC
      WRITE(6,601) (NERR(I),I=1,5),JA,JB,JC,LA,LB,LC
      GO TO 68
C  66 ENCODE(60,602,NERR(6))JA,JB,LB,LA,JC,LC
   66 WRITE(6,602) (NERR(I),I=1,5),JA,JB,LB,LA,JC,LC
   68 CALL ERROR(5,NERR)
      GO TO 70
  601 FORMAT(5A4,
     &     'D6J/DSIXJ) (',2(I4,'/2,'),I4,'/2:',2(I4,'/2,'),I4,'/2);;')
  602 FORMAT(5A4,
     &     'DRAC/DRACAH)  W(',3(I4,'/2,'),I4,'/2:',I4,'/2,',I4,'/2);;')
      END
C*DECK BCODI2
      SUBROUTINE BCODI2(NM)
      IMPLICIT REAL*8(A-H,O-Z)
      COMMON /BCODCM/Q(6400)
C   ** MAIN ROUTINE DE HITSUYO NA COMMON   CODED BY M.OKA
C   **  COMMON /BCODCM/Q(NM,NM)
C   ** Q(M,I) (I.GE.M) BINOMIAL COEF. C(I-1,M-1)
C   ** Q(I,M) (I.GE.M) Q(M,I)**(-1/2)
      INTEGER NERR(15)
      DATA NMAX/0/
      NA(I,J)=I+(J-1)*NM
C** **
      IF(NM.LT.3) GO TO 24
      IF(NMAX.EQ.NM) RETURN
      IF(NMAX.NE.0) GO TO 23
      NMAX=NM
      Q(1)=1.D0
      Q(2)=1.D0
      Q(NM+1)=1.D0
      Q(NM+2)=1.D0
      DO 10 I=3,NM
      Q(NA(I,1))=1.D0
      Q(NA(I,I))=1.D0
      Q(NA(1,I))=1.D0
      DO 10 J=2,I-1
      Q(NA(J,I))=Q(NA(J-1,I-1))+Q(NA(J,I-1))
   10 CONTINUE
      DO 20 J=3,NM
      DO 20 I=1,J-1
   21 Q(NA(J,I))=1.D0/SQRT(Q(NA(I,J)))
   20 CONTINUE
      RETURN
   22 IF(Q(NA(3,13)).EQ.66.D0)RETURN
C  23 ENCODE(57,101,NERR)NMAX,NM
   23 WRITE(6,101) NMAX,NM
      GO TO 25
C  24 ENCODE(33,100,NERR)NM
   24 WRITE(6,100) NM
   25 CALL ERROR(-48,NERR)
      RETURN
  100 FORMAT('NMAX.LT.3 IN (BCODIN)  NMAX=',I4,';')
  101 FORMAT('/BCODCM/ LENGTH MISMATCHED  OLD NMAX=',I4,', NEW NMAX=',
     +I4,';')
      END
C*DECK BCODIN
      SUBROUTINE BCODIN(NM)
      IMPLICIT REAL*8 (A-H,O-Z)
C C C C C
      COMMON /BCODCM/ Q(80,80)
C C C C C=
      IF(NM.NE.80) GO TO 99
      Q(1,1)=1.0D0
      Q(2,1)=1.0D0
      Q(1,2)=1.0D0
      Q(2,2)=1.0D0
      DO 10 I=3,80
      Q(I,1)=1.0D0
      Q(1,I)=1.0D0
      Q(I,I)=1.0D0
      DO 11 J=2,I-1
      Q(J,I)=Q(J-1,I-1)+Q(J,I-1)
   11 CONTINUE
   10 CONTINUE
      DO 20 J=3,80
      DO 20 I=1,J-1
      Q(J,I)=1.0D0/SQRT(Q(I,J))
   20 CONTINUE
      RETURN
   99 CALL BCODI2(NM)
      RETURN
      END
C*DECK FUGO
      FUNCTION FUGO(K)
      REAL*8 FUGO,FG(0:1)/1.0D0,-1.0D0/
      KK=IAND(K,1)
      FUGO=FG(KK)
      RETURN
      END
C*DECK ERROR
      SUBROUTINE ERROR(IER,NERR)
C     INTEGER NERR(*),NERR2(67),CHLENG
      INTEGER NERR(*)
C C C C C
C     ILENG=CHLENG(NERR)
C     IF(ILENG.GT.256) ILENG=256
C     CALL MOVEC(NERR2,13,NERR,1,ILENG)
C     CALL ERRSET(150,1,1)
C     NERR2(1)=ILENG+8
C     IF(IER.GT.0) IERR=150+IER
C     IF(IER.LE.0) IERR=150
C     ENCODE(8,100,NERR2(2)) IERR
C 100 FORMAT('JML',I3,'I ')
C     CALL ERRMON(NERR2,IRC,IERR)
C     IF(IERR.NE.150) RETURN
      STOP 'STOPPED IN "ERROR"'
      END
