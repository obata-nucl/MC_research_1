C    CODED AND PROGRAMED BY T. OTSUKA.
 
C    THE USER SHOULD REFER TO THE "PROGRAM AND PROGRAMMER'S NAME".
      PROGRAM  NPBTRN
C    Revised in May 1992 for FACOM/UNIX in RIKEN.
C    Subroutine BTRSET corrected on Oct. 24, 1997 by Yoshida.
C    Last revision: Nov. 01, 2000
C *****
C ***   FILE - 11  :  CFP
C ***          20  :  eigenfunction  (INPUT)
C ****
C ***          30  :  eigenfunction  (TEMPORARY  USE)
C *****
      IMPLICIT  REAL*8(A-H,O-Z)
      PARAMETER  ( LMAX=30, NEST=20, NBSCHG=1, IEX=100, INPD=6000 )
      PARAMETER  (NM=72)
      CHARACTER*1  ITYPE, IPARM, C1
      CHARACTER*9  DAY, DAYHML
      CHARACTER*8  ITM, ITMHML
      CHARACTER*20  COMCHI
      CHARACTER*3  MASS
      CHARACTER*2  NUCL
C     CHARACTER  ITIME*8,YMD*8,HMS*10,STT*5
      CHARACTER  YMD*8,HMS*10,STT*5
      COMMON / RAC    / NDMAX, NDTOP, RACTR(0:22,0:22,0:22)
      COMMON / BCODCM / Q(2*NM*NM)
      COMMON / GEN    / ENERG(NEST,0:LMAX), NNF2, NPF2, NVMAX
      COMMON / BE2    / XN, XP, BCHRGN(3), BCHRGP(3),
     &                  DQN, DQP, DQ3, RDDN, RDDP, FDDN, FDDP
      COMMON / BK     / IBK(0:LMAX,4), NDUPT(0:LMAX)
      COMMON / FINI   / JQBS(20), JSTN(INPD), JSTP(INPD),
     &                  VECJ(INPD,NEST)
      COMMON / INI    / IQBS(20), ISTN(INPD), ISTP(INPD),
     &                  VECI(INPD,NEST)
      COMMON / VERF   / IVER(10), ITITLE(2), DAYH(2), TMH, CHNHML,
     &                  CHPHML
      COMMON / EXP    / NDATA, ILIST, JXI(IEX), NXI(IEX), JXF(IEX),
     &                  NXF(IEX), BXEM(IEX), BXERR(IEX), NAVAIL(IEX)
      EQUIVALENCE      (DAYH, DAYHML), (TMH, ITMHML)
      EQUIVALENCE   ( ITITLE(1), MASS ), ( ITITLE(2), NUCL )
      DIMENSION  ITYPE(2)
C
C     OPEN(5,FILE='npbtrnin.dat',STATUS='OLD',ACCESS='SEQUENTIAL',
C    &     FORT='FORMATTED')
c      OPEN(6,STATUS='SCRATCH',ACCESS='SEQUENTIAL',FORM='FORMATTED')
      OPEN(6,ACCESS='SEQUENTIAL',FORM='FORMATTED')
      OPEN(8,FILE='out2.dat',ACCESS='SEQUENTIAL',FORM='FORMATTED')
c     OPEN(10,FILE='/home/usr/1/t940021/np/npbos/ddmefl.dat',
c    &     STATUS='OLD',ACCESS='SEQUENTIAL',FORM='UNFORMATTED')
c     OPEN(10,FILE='/home/usr/1/t940021/np/npbos/ddmefl.dat',
c    &     STATUS='OLD',ACCESS='SEQUENTIAL',FORM='UNFORMATTED')
      OPEN(10,FILE='ddmefl.dat',
     &     STATUS='OLD',ACCESS='SEQUENTIAL',FORM='UNFORMATTED')
c     OPEN(11,FILE='/home/usr/1/t940021/np/npbos/cfp2.dat',
c    &     STATUS='OLD',ACCESS='SEQUENTIAL',FORM='UNFORMATTED')
      OPEN(11,FILE='cfp2.dat',
     &     STATUS='OLD',ACCESS='SEQUENTIAL',FORM='UNFORMATTED')
      OPEN(20,FILE='npbosvec.dat',STATUS='OLD',ACCESS='SEQUENTIAL',
     &     FORM='UNFORMATTED')
      OPEN(30,STATUS='SCRATCH',ACCESS='SEQUENTIAL',FORM='UNFORMATTED')
C  to cope with REWIND trouble
C     WRITE(30) 'NPBTRN'
C     REWIND 30
C
      ITYPE(1) = 'E'
      ITYPE(2) = 'M'
      NVMAX = NEST
      NDMAX = 11




      CALL   B T R S E T
      CALL   C O M I N 1  (NDMAX)
      CALL   D D M E I N  (1)
      CALL   R D I N      (20)




      NDTOP = NNF2
      IF( NDTOP .LT. NPF2 )  NDTOP = NPF2
C  The date of running the program should be put in "DAY" in nine
C  characters.
C     DAY='         '
C  In many systems:
C     CALL   DATE ( DAY )
C  In the SUN in the education system in Kansai Univ. Takatsuki Campus:
C      CALL DATE_AND_TIME(YMD,HMS,STT,I)
      DAY=YMD(7:8)//'.'//YMD(5:6)//'.'//YMD(3:4)//' '



      CALL   ATIME( ITM )



9     READ(5,500)  MUL, IPARM, ILIST
      IF( MUL .LT. 0 )  STOP 'NORMAL TERMINATION'
 
      IF( IPARM .NE. ' ' )  THEN
        READ(5,510)  XN, XP, DQN, DQP, DQ3
        COMCHI = 'EXPLICIT  INPUT     '
      ELSE
        IF( MUL .EQ. 2 )  THEN
          XN = CHNHML
          XP = CHPHML
          COMCHI = 'SAME  AS  IN  QQ-INT'
        ELSE
          XN = 1
          XP = 1
          COMCHI = 'NO  D(+)*S + S(+)*D '
        ENDIF
      ENDIF
 
!      WRITE(8,601)  MASS, NUCL, ITYPE(IAND(MUL,1)+1), MUL,
!     &              DAY, ITM, DAYHML, ITMHML
 
      READ(5,530)  NDATA
      WRITE(8,610)  NDATA
      DO 230  I = 1, NDATA
        READ(5,530)  JXI(I), NXI(I), JXF(I), NXF(I), C1,
     &               BXEM(I), BXERR(I)
        IF( C1 .EQ. ' ' )  NAVAIL(I) = 0
        IF( C1 .NE. ' ' )  NAVAIL(I) = 1
        C1 = ' '
        IF( JXI(I) .EQ. JXF(I) .AND. NXI(I) .EQ. NXF(I) )  C1 = '*'
        IF( NAVAIL(I) .EQ. 0 )  THEN
           WRITE(8,612)  JXI(I), NXI(I), JXF(I), NXF(I), 
     &          BXEM(I), BXERR(I), C1
        ELSE
          WRITE(8,613)  JXI(I), NXI(I), JXF(I), NXF(I)
        END IF
230   CONTINUE
      WRITE(8,614)
 
      DO 2  I = 1, NBSCHG
        READ(5,501) BCHRGN(I), BCHRGP(I)
2     CONTINUE
      WRITE(8,640)  ( BCHRGN(I), BCHRGP(I), I = 1, NBSCHG )
      IF( MUL .EQ. 2 .OR. ( MUL .NE. 2 .AND. IPARM .NE. ' ' ) )
     &       WRITE(8,650)  XN, XP, COMCHI
      IF( DQN .NE. 0 .OR. DQP .NE. 0 )  WRITE(8,652)  DQN, DQP, DQ3
 
      IF( MUL .EQ. 1 )  WRITE(8,620)
      WRITE(8,630)  ITYPE(IAND(MUL,1)+1), MUL
 
1     READ(5,520) LI, LJ
      IF(LI.LT.0.OR.LJ.LT.0) GOTO 9
      IST = IBK(LI,1)
      JST = IBK(LJ,1)
      IF(IST.LE.0.OR.JST.LE.0) GOTO 1
      IEIG = IBK(LI,3)
      JEIG = IBK(LJ,3)
      IF(IEIG.LE.0.OR.JEIG.LE.0)  GOTO 1





      CALL   F I L L  (IQBS, ISTN, ISTP, VECI, LI, IST, IEIG)
      CALL   F I L L  (JQBS, JSTN, JSTP, VECJ, LJ, JST, JEIG)
 


      WRITE(8,660)  ITYPE(IAND(MUL,1)+1), MUL
 


      CALL   B E M    (LI, IST, IEIG, LJ, JST, JEIG, MUL)
 


      GOTO 1
 
500   FORMAT(I2,1X, A1, I2)
501   FORMAT(2F10.4)
510   FORMAT(8F10.4)
520   FORMAT(2I5)
530   FORMAT(2I2, 2X, 2I2, 1X, A1, 2F13.7 )
601   FORMAT('1' //T5, '*----------*'
     &           / T5, '*  ', A3, T16,'*'
     &           / T5,'*', T12,A2,T16,'*'
     &           / T5, '*----------*'
     &       //T5,'TRANSITION : ',A1,1X,I1
     &       //T5,'DATE : ',A9, 5X, 'TIME : ',A8
     &       //T10, 'INPUT-FILE -->   DATE : ',A9,5X,'TIME : ',A8)
610   FORMAT(//T5, 'EXPERIMENTAL  DATA    (# OF DATA =',I3, ')'
     &  // T11,'INITIAL', T26,'FINAL', T40,'B(EL OR ML)',
     &     T57,'ERROR'
     &  /  T11,' J  (#)', T26,'J  (#)' )
612   FORMAT( T11,I2, T14,'(', I2,')', T25,I2, T28,'(',
     &        I2,')', T39,F11.4, T53,F10.4, 3X, A1 )
613   FORMAT( T11,I2, T14,'(', I2,')', T25,I2, T28,'(',
     &        I2,')', T39,'NO DATA (CALCULATED IN ILIST=0 MODE)')
614   FORMAT(/ T15,'DATA WITH * IS SUPPOSED TO BE MOMENT OR G-FACTOR')
620   FORMAT(  T8,'M 1  -->  THESE ARE BOSON G-FACTORS' )
630   FORMAT( //T2, '*** CALCULATED  &  EXPERIMENTAL  ',A1,' ',I1,
     &          '  MATRIX  ELEMENTS  (OR  B(EL OR ML)''S) ***'
     &        //T15, '* ON LAST COLUMN INDICATES MOMENT OR G-FACTOR',
     &          '  (M1 = G*L(TOT))'
     &        / T15, '@ ON LAST COLUMN INDICATES EXPERIMENTAL DATA' )
640   FORMAT( //T5,'BOSON  TRANSITION  PARAMETERS'
     &        //T25,'NEUTRON', T40,'PROTON'
     &        //T5,'BOSON  CHARGE', T25,F7.3, T39,F7.3
     &        / (T25,F7.3, T39,F7.3 / ) )
650   FORMAT(   T5,'CHI (DS / DD)', T25,F7.3, T39,F7.3, T52,A20 )
652   FORMAT(  /T5,'DQ (Q(T)*ND(NOT.T))   N :',F8.4, 4X,'P :',F8.4
     &         /T5,'DQ3 (/(1-DQ3*NDN*NDP)) :',F10.6 )
660   FORMAT(/T2, '***',19('****')
     &   //T3,'LI', T10,'LF', T18,'DS (N)', T26,'DD (N)',
     &     T35,'DS (P)', T43,'DD (P)', T51,'NEUTRON', T60,'PROTON',
     &     T69,'B(',A1,I1,';LI->LF)' / )
      END









      SUBROUTINE   B E M  ( LI, IST, IEIG, LJ, JST, JEIG, MUL )
C***********************************************************************
C************        < J !!  Q(N)  &  Q(P)  !! I >      ****************
C***********************************************************************
      IMPLICIT  REAL*8  (A-H,O-Z)
      PARAMETER   (ICFP1=2907, ICFP2=6371)
      PARAMETER   ( LMAX=30, NEST=20, NBSCHG=1, IEX=100, INPD=6000 )
      CHARACTER*1   MOMIND
C **********************************************************************
      COMMON / RAC    / NDMAX, NDTOP, RACTR(0:22,0:22,0:22)
C **********************************************************************
      COMMON / GEN    / ENERG(NEST,0:LMAX), NNF2, NPF2, NVMAX
      COMMON / BE2    / XN, XP, BCHRGN(3), BCHRGP(3),
     &                  DQN, DQP, DQ3, RDDN, RDDP, FDDN, FDDP
      COMMON / BK     / IBK(0:LMAX,4), NDUPT(0:LMAX)
      COMMON / INI    / IQBS(20),ISTN(INPD),ISTP(INPD),VECI(INPD,NEST)
      COMMON / FINI   / JQBS(20),JSTN(INPD),JSTP(INPD),VECJ(INPD,NEST)
      COMMON / TRNS   / XIJ(8), TI(NEST,8), BE(NEST,NEST,8)
      COMMON / SRCM   / SR(0:20), SRJK(0:50)
      COMMON / PH     / PHASE(0:1)
      COMMON / SPSTCM / ISTBS(300),NBD(300),LD(300),NSEN(300),NDL(300),
     &                  NBBAS(16),ISTBS2(300),LAD(300),LADBS(16)
C--------------------------------------------------------------------
C   ISTBS(I) =(BASE ADDRESS-1) FOR PARENT STATE I IN CFP1 ARRAY
C   ISTBS2(I)= SAME AS ISTBS , BUT FOR CFP2 ARRAY
C   NBD(I)   = # OF D-BOSONS IN STATE I
C   LD(I)    = L-VALUE FOR STATE I
C   NSEN(I)  = SENIORITY FOR STATE I
C   NDL(I)   = OTHER QUANTUM = TO SPECIFY STATE I (=0,1)
C   NBBAS(I) = LAST STATE= WITH (K-1) D-BOSONS
C--------------------------------------------------------------------
      COMMON / CFP1CM / IDAU(ICFP1), L1(ICFP1), CFP1(ICFP1)
      COMMON / CFP2CM / IDAU2(ICFP2), L2(ICFP2), CFP2(ICFP2), LIM(ICFP2)
C--------------------------------------------------------------------
C   IDAU  = DAUGHTER STATE = (HAS LESS BOSONS THEN PARENT STATE)
C   L1    = L-VALUE OF DAUGHTER STATE
C   CFP1  = VALUE OF 1-BODY CFP
C   CFP2  = VALUE OF 2-BODY CFP
C   LIM    = L-VALUE OF THE TWO DECOUPLED BOSONS FOR CFP2
C--------------------------------------------------------------------
      COMMON / EXP    / NDATA, ILIST, JXI(IEX), NXI(IEX), JXF(IEX),
     &                  NXF(IEX), BXEM(IEX), BXERR(IEX), NAVAIL(IEX)
      DIMENSION BETW(3)
      IAND1(I) = IAND(1,I)
      NOP = 8
      IF( MUL .NE. 2 )  NOP = 2
      NDF = 1
        IF( MUL .NE. 2 )  NDF = 0
      IOP = 1
        IF( MUL .NE. 2 )  IOP = 2
      DO 101  K = 1, NEST
      DO 101  L = 1, NEST
      DO 101  M = 1, 8
        BE(K,L,M) = 0
101   CONTINUE
C-----------------------------------------------------------------------
      MAXL = NDTOP*2
      DO 210  I3 = 0, MAXL
      DO 210  I2 = 0, MAXL
      DO 210  I1 = 0, MAXL
        RACTR(I1, I2, I3) = DRAC( I1*2, I2*2, MUL*2, LI*2, LJ*2, I3*2 )
210   CONTINUE
C-----------------------------------------------------------------------
      DO 110  I = 1, IST
        IN=ISTN(I)
        IP=ISTP(I)
        NNBI=NBD(IN)
        FDN = NNBI
        NPBI=NBD(IP)
        FDP = NPBI
        LIN=LD(IN)
        LIP=LD(IP)
        DO 102  M = 1, 8
        DO 102  L = 1, JEIG
          TI(L,M) = 0
102     CONTINUE
        NDI = NNBI + NPBI
        IF( NDI .LE. 1 )  J1 = 1
        IF( NDI .GE. 2 )  J1 = JQBS( NDI-1 ) + 1
        IF( NDI+1 .GT. NDUPT(LJ) )  J2 = JQBS( NDUPT(LJ) + 1 )
        IF( NDI+1 .LE. NDUPT(LJ) )  J2 = JQBS( NDI + 2 )
C-----------------------------------------------------------------------
        DO 120  J = J1, J2
          DO 100  M = 1, 8
            XIJ( M ) = 0
100       CONTINUE
          JN=JSTN(J)
          JP=JSTP(J)
          NNBJ=NBD(JN)
          NPBJ=NBD(JP)
          NNBD=NNBI-NNBJ
          NPBD=NPBI-NPBJ
          LJN=LD(JN)
          LJP=LD(JP)
          IF( IABS(NNBD)    .GT. NDF )  GOTO  120
          IF( IABS(NPBD)    .GT. NDF )  GOTO  120
          IF( IABS(LIN-LJN) .GT. MUL )  GOTO  120
          IF( IABS(LIP-LJP) .GT. MUL )  GOTO  120
          IF( NNBD )  91, 92, 93
 
91        IF(IP-JP) 120, 10, 120
93        IF(IP-JP) 120, 20, 120
 
92        IF(IN.NE.JN) GOTO 94
          IF(NPBD) 30,60,40
 
C      NOTE : AFTER LABEL '60' : 'GOTO 94'
 
94        IF(IP-JP) 99,50,99
C ========================================
C      CALCULATE < N !! D(+)*S !! N-1 >
C ========================================
10        KL = ISTBS(JN) + 1
          KR = ISTBS(JN+1)
          DO 11 K=KL,KR
C      FIND DAUGHTER STATE
            IF(IDAU(K).EQ.IN) GOTO 12
11        CONTINUE
C      DAUGHTER STATE 'IN' NOT FOUND
          GOTO 99
C      DAUGHTER STATE 'IN' FOUND
12        XIJ(1) = PHASE(IAND1(LJN+LIN)) * RACTR( LJN, LJP, LIN )
     &           * CFP1(K) * SRJK(LJN) * SRJK(LJ) * SRJK(LI)
     &           * SR(NNBJ) * SR(NNF2-NNBI)
          XIJ(5) = XIJ(1) * FDP
          GOTO 99
C -------------
C      CALCULATE < N-1 !! (S+)*D !! N >
20        KL = ISTBS(IN) + 1
          KR = ISTBS(IN+1)
          DO 21  K = KL, KR
            IF(IDAU(K).EQ.JN) GOTO 22
21        CONTINUE
          GOTO 99
22        XIJ(1) = RACTR( LJN, LJP, LIN ) * CFP1(K)
     &             * SRJK(LIN) * SRJK(LJ) * SRJK(LI)
     &             * SR(NNBI) * SR(NNF2-NNBJ)
          XIJ(5) = XIJ(1) * FDP
          GOTO 99
C -------------
C      CALCULATE < P !! (D+)*S !! P-1 >
30        KL = ISTBS(JP) + 1
          KR = ISTBS(JP+1)
          DO 31 K=KL,KR
            IF(IDAU(K).EQ.IP) GOTO 32
31        CONTINUE
          GOTO 99
32        XIJ(3) = PHASE(IAND1(LJP+LIP)) * RACTR( LJP, LJN, LIP )
     &             * CFP1(K) * SRJK(LJP) * SRJK(LJ) * SRJK(LI)
     &             * SR(NPBJ) * SR(NPF2-NPBI)
          IF( IAND(LJP+LI+LJ+LIP,1) .EQ. 1 )  XIJ(3) = - XIJ(3)
          XIJ(7) = XIJ(3) * FDN
          GOTO 99
C -------------
C      CALCULATE  < P-1 !! (S+)*D !! P >
40        KL = ISTBS(IP) + 1
          KR = ISTBS(IP+1)
          DO 41 K=KL,KR
            IF(IDAU(K).EQ.JP) GOTO 42
41        CONTINUE
          GOTO 99
42        XIJ(3) = RACTR( LJP, LJN, LIP ) * CFP1(K)
     &             * SRJK(LIP) * SRJK(LJ) * SRJK(LI)
     &             * SR(NPBI) * SR(NPF2-NPBJ)
          IF( IAND(LJP+LI+LJ+LIP,1) .EQ. 1 )  XIJ(3) = - XIJ(3)
          XIJ(7) = XIJ(3) * FDN
          GOTO 99
C -------------
C      CALCULATE   < N !! (D+)*D !! N >
   50 XIJ( 2 ) = RACTR( LJN, LJP, LIN ) * SRJK(LJ) * SRJK(LI)
     1           * DDMAT( JN, IN, MUL )
      XIJ(6) = XIJ(2) * FDP
      GOTO 99
 
C      CALCULATE  < P !! (D+)*D !! P >
   60 XIJ( 4 ) = RACTR( LJP, LJN, LIP ) * SRJK(LJ) * SRJK(LI)
     1           * DDMAT( JP, IP, MUL )
      IF( IAND(LJP+LI+LJ+LIP,1) .EQ. 1 )  XIJ(4) = - XIJ(4)
      XIJ(8) = XIJ(4) * FDN
      GOTO 94
 
   99 CONTINUE
 
C       XIJ(1) = <J !! (D+*S+S+*D) N !! I>
C       XIJ(2) = <J !! ((D+)*D)(2) N !! I>
C       XIJ(3) = <J !! (D+*S+S+*D) P !! I>
C       XIJ(4) = <J !! ((D+)*D)(2) P !! I>
 
        DO 122  L =  1, JEIG
        DO 122  MM = 1, NOP
          M = MM * IOP
          TI(L,M)=TI(L,M)+XIJ(M)*VECJ(J,L)
122     CONTINUE
 
120   CONTINUE
C
        DO 112  L =  1, JEIG
        DO 112  MM = 1, NOP
          M = MM * IOP
          TI(L,M) = TI(L,M) / ( 1 - DQ3 * FDN * FDP )
112     CONTINUE
 
        DO 114 K=1,IEIG
        DO 114 L=1,JEIG
        DO 114  MM = 1, NOP
          M = MM * IOP
          BE(K,L,M) = BE(K,L,M) + TI(L,M) * VECI(I,K)
114     CONTINUE
 
110   CONTINUE
 
C-----------------------------------------------------------------------
 
      IF( LI .GE. 1 )  THEN
        FM1 = SQRT( 10.D0 / (LI*(LI+1)*(2*LI+1)) )
      ENDIF
      DO 130  K = 1, IEIG
      DO 131  L = 1, JEIG
        IF( LI .EQ. LJ .AND. K .GT. L )  GOTO  131
        BEN = BE(K,L,1) + XN*BE(K,L,2) + DQN*(BE(K,L,5)+XN*BE(K,L,6))
        BEP = BE(K,L,3) + XP*BE(K,L,4) + DQP*(BE(K,L,7)+XP*BE(K,L,8))
        DO 133  I = 1, NBSCHG


          IF( ( MUL .EQ. 1  .OR.  MUL .EQ. 2 ) .AND.
     &        ( LI  .EQ. LJ .AND. K   .EQ. L ) )  THEN
            FLI=LI
            IF(MUL.EQ.1) BETW(I) = (BCHRGN(I)*BEN+BCHRGP(I)*BEP)
     &                   * FM1
            IF(MUL.EQ.2) BETW(I) = SQRT( ( 16*3.141593D0 * LI
     &          * (2*LI-1) )/ ( 5 * (FLI+1) * (2*LI+1) * (2*LI+3) ) )
     &         * (BCHRGN(I)*BEN+BCHRGP(I)*BEP)
c***  on 12/2, 2010  ***
            write(8,*) BETW(I)
c***
          ELSE
            BETW(I) = ( BCHRGN(I)*BEN + BCHRGP(I)*BEP )**2 / (2*LI+1)
            IF( MUL .EQ. 1 )  BETW( I ) = BETW( I ) * 3 * 10
     &                                  / ( 4 * 3.141593D0 )
          ENDIF



133   CONTINUE
 
      IF( ILIST .GE. 1 )  GOTO  70
        DO 202  I = 1, NDATA
          IF(  (JXI(I) .EQ. LI .AND. NXI(I) .EQ. K .AND.
     &          JXF(I) .EQ. LJ .AND. NXF(I) .EQ. L )  .OR.
     &         (JXF(I) .EQ. LI .AND. NXF(I) .EQ. K .AND.
     &          JXI(I) .EQ. LJ .AND. NXI(I) .EQ. L )  )    GOTO  70
202     CONTINUE
        GOTO  131
 
70      MOMIND = ' '
        IF( (LI .EQ. LJ) .AND. (K .EQ. L) )  MOMIND = '*'
 
        IF( MUL .EQ. 1 )  THEN
          IF( LI .EQ. LJ .AND. K .EQ. L )  THEN
            BEN = BEN * FM1
            BEP = BEP * FM1
          ELSE
            BEN = BEN * SQRT( 30.D0 / ( 4 * 3.141593D0 ) )
            BEP = BEP * SQRT( 30.D0 / ( 4 * 3.141593D0 ) )
          ENDIF
        ENDIF
 
        WRITE(8,600)  LI, K, LJ, L, ( BE(K,L,M), M = 1, 4 ),
     &                BEN, BEP, (BETW(I), I = 1, NBSCHG ), MOMIND
 
        IF( DQN .NE. 0 .OR. DQP .NE. 0 )  THEN
          WRITE(8,602)  ( BE(K,L,M), M = 5, 8 )
        ENDIF
 
        DO 200  I = 1, NDATA
            IF(  JXI(I) .EQ. LI .AND. NXI(I) .EQ. K .AND.
     &           JXF(I) .EQ. LJ .AND. NXF(I) .EQ. L .AND.
     &           NAVAIL(I) .EQ. 0 )
     &             WRITE(8,610)  BXEM(I), BXERR(I)
          IF( MOMIND .EQ. '*' )  GOTO 200
 
            IF(  JXF(I) .EQ. LI .AND. NXF(I) .EQ. K .AND.
     &           JXI(I) .EQ. LJ .AND. NXI(I) .EQ. L .AND.
     &           NAVAIL(I) .EQ. 0 )     THEN
              X = ( BXEM (I) * ( 2*LJ + 1 ) ) / ( 2*LI + 1 )
              Y = ( BXERR(I) * ( 2*LJ + 1 ) ) / ( 2*LI + 1 )
              WRITE(8,610)  X, Y
            ENDIF
200     CONTINUE
 
131   CONTINUE
130   CONTINUE
 
      RETURN
600   FORMAT( T2,I2, T5,I2, T5,'(', T7,')', T9,I2, T12,I2, T12,'(',
     &        T14,')', T16,4F8.3,
     &        T49,'*',2F8.3, T67,'*', F11.4, 1X,A1 )
602   FORMAT( T7,'Q * ND :', T16, 4F8.3 )
610   FORMAT( T55, 'EXP DATA --> ', F11.4, T80, '@'
     &      / T59, '( ERROR =',     F11.4, T80, ')' )
      END











      SUBROUTINE   R D I N  (IO)
      IMPLICIT  REAL*8(A-H,O-Z)
      PARAMETER   ( LMAX=30, NEST=20, LFL=30, INPD=6000 )
      CHARACTER   IVER*4, ITITLE*4, DAYV*8, TMV*8
      COMMON / INI  / NQBS(20), ISTN(INPD), ISTP(INPD), VEC(NEST*INPD)
      COMMON / GEN  / ENERG(NEST,0:LMAX), NNF2, NPF2, NVMAX
      COMMON / BK   / IBK(0:LMAX,4), NDUPT(0:LMAX)
      COMMON / VERF / IVER(10), ITITLE(2), DAYV(2), TMV, CHHML(2)
C ***************************************
C **    IBK(LP,1) : NST
C **    IBK(LP,2) : NXC
C **    IBK(LP,3) : NEIG
C **    IBK(LP,4) : # OF RECORDS ON FILE-LFL TO BE SKIPPED
C ***************************************
C     REWIND  IO
      DO 1  J = 1, 4
      DO 1  I = 0, LMAX
        IBK( I, J ) = 0
1     CONTINUE
C----------------------------
C       FIRST RECORD
C----------------------------
      ISKIP = 0
      REWIND   LFL
    4 READ(IO,END=99) I, J, K, L, NNF2, NPF2
 
      IF(I.LT.0.OR.L.GE.LMAX) GOTO 99
      NDUPT( L ) = I
      IBK( L, 4 ) = ISKIP
C----------------------------
C       SECOND RECORD
C----------------------------
      READ(IO)  ICODE, IVER, ITITLE, DAYV, TMV, CHHML
C----------------------------
C       THIRD RECORD
C----------------------------
      READ(IO)  NXC, (NQBS(I),I=1,NXC)
 
      WRITE( LFL )  ( NQBS(I), I = 1, NXC )
 
      IBK( L, 2 ) = NXC
C----------------------------
C      FOURTH RECORD
C----------------------------
      READ(IO) NST,(ISTN(I),I=1,NST),(ISTP(I),I=1,NST)
 
      WRITE( LFL )  ( ISTN(I), I = 1, NST )
      WRITE( LFL )  ( ISTP(I), I = 1, NST )
 
      IBK( L, 1 ) = NST
      ISKIP = ISKIP + 3
C----------------------------
C       FIFTH RECORD
C----------------------------
      NEIG=0
2     READ(IO) K, E
      IF( K .LE. 0 ) GOTO  3
C----------------------------
C       SIXTH RECORD
C----------------------------
      READ(IO)  ( VEC(I), I = 1, NST )
 
      IF( NEIG .GE. NVMAX )  GOTO  2
      NEIG = NEIG + 1
      ENERG( NEIG, L ) = E
 
      WRITE( LFL )  ( VEC(I), I = 1, NST )
 
      ISKIP = ISKIP + 1
 
      GOTO 2
3     IBK( L, 3 ) = NEIG
      GOTO 4
C--------------------------------------------
C      NEGATIVE NUMBER IN FIRST RECORD
C--------------------------------------------
99    CONTINUE
      REWIND  LFL
      CALL   F C L O S E  ( IO )
      RETURN
      END








      SUBROUTINE   F I L L  ( NQBS, NSTN, NSTP, VEC, LP, NST, NEIG )
      IMPLICIT  REAL*8  (A-H, O-Z)
      PARAMETER   ( LMAX=30, LFL = 30, INPD=6000)
      COMMON / BK     / IBK(0:LMAX,4), NDUPT(0:LMAX)
      DIMENSION   NQBS(1), NSTN(1), NSTP(1), VEC(INPD,*)
      NXC = IBK( LP, 2 )
 
      DO 200  I = 1, IBK( LP, 4 )
        READ( LFL )
200   CONTINUE
 
      READ( LFL )  ( NQBS(I), I = 1, NXC )
 
      READ( LFL )  ( NSTN(I), I = 1, NST )
      READ( LFL )  ( NSTP(I), I = 1, NST )
 
      DO 100  K = 1, NEIG
        READ( LFL )  ( VEC( I, K ), I = 1, NST )
100   CONTINUE
 
      REWIND  LFL
      RETURN
      END








      SUBROUTINE   B T R S E T
      IMPLICIT  REAL*8(A-H,O-Z)
      PARAMETER  ( LMAX=30 )
      PARAMETER  (NM=72)
      COMMON / PH   / PHASE(0:1)
      COMMON / SRCM / SR(0:20), SRJK(0:50)
      COMMON/ BCODCM /Q(NM,NM),P(NM,NM)
      COMMON / BE2  / XN, XP, BCHRGN(3), BCHRGP(3), XX(7)
      COMMON / BK   / IBK(0:LMAX,4), NDUPT(0:LMAX)
      COMMON / SR / SQR(501)
      PHASE(0) =  1
      PHASE(1) = -1
      DO 200  I = 0 , 20
  200 SR(I) = DSQRT( DFLOAT(I) )
      DO 210  I = 0, 50
  210 SRJK(I) = DSQRT( DFLOAT(2*I+1) )
      DO 220  J = 1, NM
      DO 220  I = 1, NM
  220 Q(I,J) = 0
C     C A L L     B C O D I N (52)
      C A L L     B C O D I N
      DO 260  J = 1, 7
260   XX( J ) = 0
C     DO 230  J = 1, 6          corrected on Oct. 24, 1997 by Yoshida
      DO 230  J = 1, 4
C     DO 230  I = 0, 30         revised on Oct. 24, 1997 by Yoshida
      DO 230  I = 0, LMAX
  230 IBK (I,J) = 0
C     DO 240  I = 1, 90         corrected on Oct. 24, 1997 by Yoshida
      DO 240  I = 0, LMAX
  240 NDUPT ( I ) = 0
      SQR(1) = 0
      DO 250  I = 1, 500
  250 SQR( I + 1 ) = DSQRT( DFLOAT( I ) )
      RETURN
      END









      SUBROUTINE    A T I M E   ( ITIME )
      CHARACTER*8 ITIME
C     ITIME='        '
C***********************************
C*****    S U N    VERSION     *****
C-----------------------------------
C      CHARACTER  ITIME*8,YMD*8,HMS*10,STT*5
C      CALL DATE_AND_TIME(YMD,HMS,STT,I)
C      ITIME=HMS(1:2)//':'//HMS(3:4)//':'//HMS(5:6)
C***********************************
C*****    V A X    VERSION     *****
C-----------------------------------
C     CHARACTER*8  ITIME
C     CALL   T I M E  ( ITIME )
C***********************************
C*****   F A C O M  VERSION    *****
C-----------------------------------
C     CHARACTER*8  ITIME
C     CALL   T I M E  ( ITX )
C     ITX = ITX / 1000
C     ITH = ITX / 3600
C     ITM = ( ITX - ITH * 3600 ) / 60
C     ITS = ( ITX - ITH * 3600 - ITM * 60 )
C     WRITE( ITIME, 900 )  ITH, ITM, ITS
C900  FORMAT( I2, ':', I2, ':', I2 )
C***********************************
C*****   H I T A C  VERSION    *****
C-----------------------------------
c     CHARACTER*12 ITIMEX
c     CHARACTER*8  ITIME
c     CALL   C L O C K ( ITIMEX, 1 )
c     WRITE( ITIME, 900 )  ITIMEX
c900  FORMAT( A8 )
      RETURN
      END




C     SUBROUTINE SECOND(T)
C
C TIMER ROUTINE TO RETURN CPU TIME IN SECONDS.
C EITHER A CALL TO ZERSEC OR THE INITIAL CALL TO SECOND
C WILL ZERO THE TIME, AND SUBSEQUENT CALLS TO SECOND
C WILL GIVE TIMES RELATIVE TO THIS TIME.
C
C     DATA   ISET  / 0 /
C     DATA   ITINI / 0 /
C     T = 0
C*********************************************
C**    FOR  VAX   ( CODED BY K. SCHMIDT )   **
C---------------------------------------------
C     DATA   IHANDL / 0 /
C     IF (IHANDL.EQ.0) THEN
C     I=LIB$INIT_TIMER(IHANDL)
C     ITIME=0
C     ELSE
C     I=LIB$STAT_TIMER(2,ITIME,IHANDL)
C     ENDIF
C     T=ITIME/100.
C*********************************************
C*********************************************
C**    FOR  FACOM
C---------------------------------------------
C     IF( ISET .EQ. 0 )  THEN
C       CALL    CLOCK( ITINI )
C       T = 0
C       ISET = 1
C     ELSE
C       CALL    CLOCK( ITM )
C       T = ITM - ITINI
C     ENDIF
C*********************************************
C*********************************************
C**    FOR  HITAC
C---------------------------------------------
 
c     IF( ISET .EQ. 0 )  THEN
c       T = 0
c       CALL    CLOCK
c       ISET = 1
c     ELSE
c       CALL    CLOCK( T, 5 )
c     ENDIF
c
C     RETURN
 
C     ENTRY ZERSEC
C*********************************************
C**    FOR  VAX   ( CODED BY K. SCHMIDT )   **
C---------------------------------------------
C     IHANDL=0
C     I=LIB$INIT_TIMER(IHANDL)
C*********************************************
C*********************************************
C**    FOR  FACOM
C---------------------------------------------
C       CALL  CLOCK( ITINI )
C*********************************************
C*********************************************
C**    FOR  HITAC
C---------------------------------------------
c       CALL  CLOCK
c
c     ISET = 1
c     T = 0
C     RETURN
C     END







      SUBROUTINE  F C L O S E  ( IFL )
C********************************************
C
C   THIS  SUBROUTINE  CLOSES  FILE - #IFL.
C
C********************************************
C**    FOR  VAX
C-----------------------------------
C     CALL   C L O S E  ( IFL )
C***********************************
C***********************************
C**    FOR  FACOM
C-----------------------------------
      CLOSE  ( IFL )
      RETURN
      END






      FUNCTION   R A C A R  ( LA, LB, LC, LD )
      IMPLICIT REAL*8  (A-H,O-Z)
      PARAMETER  ( IDRAC=17480 )
      COMMON / RCLCM / LDMAXI, LT, LTMAXI, ISQDIM, IADRAB(120),
     &                 RACL(IDRAC)
      COMMON / SR / SR(0:500)
      COMMON / LM / MAXDIM, NMAT, MAXST, IFL, NDRAC, LVEC
      DATA   MUL / 2 /
      DIMENSION  MX(4)
      IF( LT .EQ. 0 ) THEN
        IF( LA .EQ. LB .AND. LC .EQ. LD )  THEN
          X = 1 / ( SR(2*LA+1) * SR(2*LC+1) )
          IF( IAND( LA + LC, 1 ) .EQ. 1 )  X = - X
          RACAR = X
        ELSE
          RACAR = 0
        ENDIF
        RETURN
      ENDIF
 
      IF( LT .GT. LTMAXI )  THEN
        RACAR = DRAC( LA*2, LB*2, LC*2, LD*2, LT*2, MUL*2 )
        RETURN
      ENDIF
 
      IF( (LA .GT. LDMAXI) .OR. (LB .GT. LDMAXI) .OR.
     &    (LC .GT. LDMAXI) .OR. (LD .GT. LDMAXI) )  THEN
        RACAR = DRAC( LA*2, LB*2, LC*2, LD*2, LT*2, MUL*2 )
        RETURN
      ENDIF
 
      MX(1) = LDMAXI * LA + LB
      MX(2) = LDMAXI * LB + LA
      MX(3) = LDMAXI * LC + LD
      MX(4) = LDMAXI * LD + LC
 
      MMIN = MX(1)
      IMIN = 1
      DO 200  I = 2, 4
        IF( MX(I) .GE. MMIN )  GOTO  200
        MMIN = MX(I)
        IMIN = I
200   CONTINUE
      GOTO  ( 10, 20, 30, 40 ), IMIN
 
10    IA = LA
      IB = LB
      IC = LC
      ID = LD
      GOTO  50
 
20    IA = LB
      IB = LA
      IC = LD
      ID = LC
      GOTO  50
 
30    IA = LC
      IB = LD
      IC = LA
      ID = LB
      GOTO  50
 
40    IA = LD
      IB = LC
      IC = LB
      ID = LA
 
50    IF( IA .EQ. 0 )  THEN
        IF( IC .EQ. MUL .AND. IB .EQ. LT .AND.
     &      (ID+IC) .GE. LT .AND. IABS(ID-IC) .LE. LT .AND.
     &      (ID+LT) .GE. IC .AND. IABS(ID-LT) .LE. IC )  THEN
          RACAR = 1 / ( SR(2*LT+1) * SR(2*MUL+1) )
        ELSE
          RACAR = 0
        ENDIF
 
      ELSE
        MADR = ( (IA-2) * (2*LDMAXI-IA+1) ) / 2 + IB - IA + 1
        ISQ  = (LDMAXI-1) * (IC-IA) + ID - IB + 1 + IADRAB( MADR )
        RACAR = RACL( ISQ )
 
      ENDIF
 
      RETURN
      END









      FUNCTION    D R A C L M   ( LN, LP, LDN, LDP,  LT, M )
      IMPLICIT REAL*8  (A-H,O-Z)
      COMMON / SR / SR(0:500)
 
      IF( M .EQ. 2 )  GOTO  20
      IF( M .EQ. 0 )  GOTO  10
 
      DRACLM = DRAC( LN*2, LP*2, LDN*2, LDP*2, LT*2, M*2 )
      RETURN
 
10    DRACLM = 0
      IF( LN .NE. LDN .OR. LP .NE. LDP )  RETURN
      DRACLM = 1 / ( SR(LN*2+1) * SR(LP*2+1) )
      IF( IAND( LT+LN+LP, 1 ) .EQ. 1 )  DRACLM = - DRACLM
      RETURN
 
20    DRACLM = RACAR( LN, LP, LDN, LDP )
      RETURN
      END
      SUBROUTINE   R A C I N  ( LT )
      IMPLICIT REAL*8  (A-H,O-Z)
      CHARACTER*9  FDTI
      PARAMETER  ( IDRAC=17480 )
      COMMON / RCLCM / LDMAXI, LTC, LTMAXI, ISQDIM, IADRAB(120),
     &                 RACL(IDRAC)
      COMMON / LM / MAXDIM, NMAT, MAXST, IFL, NDRAC, LVEC
      DATA  ISET / 0 /
      LTC = LT
      IF( LT .EQ. 0 )  RETURN
      IF( ISET .EQ. 1 )  GOTO  10
        READ( 9 )  LDMAXI, LTMAXI, ISQDIM, FDTI
        READ( 9 )  ( IADRAB( I ), I = 1, ISQDIM )
        NDMAXI = LDMAXI / 2
        WRITE(6,610)  NDMAXI, LTMAXI, FDTI
        IF( NDRAC .LT. NDMAXI )  STOP  'ND(MAX) ON FILE-9 : TOO LARGE'
        NDRAC = NDMAXI
        ISET = 1
      GOTO  40
10    READ( 9 )
      READ( 9 )
40    IF( LT .GT. LTMAXI )  GOTO  70
50    READ( 9 )  LTZ
      IF( LTZ .EQ. LT )  GOTO  60
      IF( LTZ .LT. 0  )  GOTO  70
      READ( 9 )
      GOTO  50
60    READ( 9 )  ( RACL( I ), I = 1, ISQDIM )
 
      REWIND  9
      RETURN
 
70    WRITE(6,650)  LT
      REWIND  9
      RETURN
610   FORMAT(//T7,'**  FILE-9  READ-IN : MAX(ND) =',I2,'   MAX(LT) =',
     &   I3,'   CALC.  DATE =',A9 )
650   FORMAT(//T7,'**  REQUESTED  LT =',I3,'  NOT  FOUND  ON  FILE'/)
      END
      SUBROUTINE   D D M E I N  ( IWDDME )
      IMPLICIT  REAL*8  (A-H, O-Z)
      PARAMETER   (IDDM=7594)
      COMMON / SPSTCM /  ISTBS(300),NBD(300),LD(300),NSEN(300),NDL(300)
     &                  ,NBBAS(16),ISTBS2(300),LAD(300),LADBS(16)
      COMMON / DDMECM /  MAXDD,NDBST,NDBST1,NMEDD,
     &                   IBSDD(300,3),ILDDM(IDDM),DDMAT(IDDM)
      COMMON / SR /  SR0,SR(500)
C     REWIND  10
      READ( 10 ) MAXDD, NDBST, NMEDD
      IF(IWDDME.GE.1)  WRITE(6,600) MAXDD, NDBST, NMEDD, IWDDME
      NDBST1 = NDBST + 1
      DO 240  K = 2, 4
      IF(IWDDME.GE.2)  WRITE(6,602)  K
      READ ( 10 )  (IBSDD(I,K-1), I=1,NDBST1)
      IF(IWDDME.GE.2)  WRITE(6,610)  (I,IBSDD(I,K-1), I=1,NDBST1)
C ***
      J1 = IBSDD(1,K-1) + 1
      J2 = IBSDD(NDBST+1,K-1)
      IF(IWDDME.GE.2)  WRITE(6,604)  K, J1, J2
      READ ( 10 )  (ILDDM(J),DDMAT(J),J=J1,J2)
      IF(IWDDME.GE.2)  WRITE(6,620)  (J,ILDDM(J),DDMAT(J),J=J1,J2)
  240 CONTINUE
      CALL   F C L O S E ( 10 )
      RETURN
  600 FORMAT(//T6,'**  D-D M.E.  INPUT  FROM  FILE-10  **'
     &   / T10,'MAXDD =',I6,5X,'NDBST =',I6,5X,'NMEDD =',I8,
     &     5X,'PRINT =',I2)
  602 FORMAT(//T6,'**  K =',I2,'  ME#  BASE  ADDRESS  FOR  EACH  STATE')
  604 FORMAT(//T6,'**  K =',I2,'  (ME#, BRA)  DD-ME   ME# =',I7,
     &   ' -->',I7 )
  610 FORMAT(/ (T9,10 ( '  (', I3, ')', I5 )) )
  620 FORMAT(/ (T9, 6 ( '  (',I4,'/',I3,')',F7.3)) )
      E N D













      FUNCTION   D D M A T  ( JL, JR, K )
      IMPLICIT  REAL*8  (A-H, O-Z)
      PARAMETER   (IDDM=7594)
      COMMON / SPSTCM /  ISTBS(300),NBD(300),LD(300),NSEN(300),NDL(300)
     1                  ,NBBAS(16),ISTBS2(300),LAD(300),LADBS(16)
      COMMON / DDMECM /  MAXDD,NDBST,NDBST1,NMEDD,
     &                   IBSDD(300,3),ILDDM(IDDM),DDMAR(IDDM)
      COMMON / SR /  SR0,SR(500)
 
      DDMAT = 0
      IF( NBD(JL) .NE. NBD(JR) )  RETURN
      IF( K .GE. 2 )  GOTO  10
      IF( JL .NE. JR )  RETURN
      IF( K .EQ. 1 )  GOTO  20
C***********************************************************************
C ***   K = 0     //     ND = SQRT( 5 ) * { D(+) D(-) }(0)
C***********************************************************************
      DDMAT = ( NBD( JL ) * SR( LD(JL)*2+1 ) ) / SR( 5 )
      RETURN
C***********************************************************************
C ***   K = 1     //     ND = SQRT( 5 ) * { D(+) D(-) }(0)
C***********************************************************************
   20 DDMAT = DSQRT( DFLOAT(LD(JL)*(LD(JL)+1)*(2*LD(JL)+1)) ) / SR(10)
      RETURN
C***********************************************************************
C ***   K = 2, 3, 4
C***********************************************************************
   10 IF(      IABS(LD(JL)-LD(JR)) .GT. K
     &    .OR.     (LD(JL)+LD(JR)) .LT. K )  RETURN
 
      IF( NBD( JL ) .GT. MAXDD )  GOTO  50
 
      IF( JL .LE. JR )  GOTO  40
      IBS1 = IBSDD( JL, K - 1 ) + 1
      IBS2 = IBSDD( JL + 1, K - 1 )
      DO 34  IX = IBS1, IBS2
      IF( ILDDM( IX ) .EQ. JR )  GOTO  36
   34 CONTINUE
      STOP 00034
   36 DDMAT = DDMAR( IX )
      IF( IAND( LD(JL)+LD(JR), 1 ) .EQ. 1 )  DDMAT = - DDMAT
      RETURN
   40 IBS1 = IBSDD( JR, K - 1 ) + 1
      IBS2 = IBSDD( JR + 1, K - 1 )
      DO 44  IX = IBS1, IBS2
      IF( ILDDM( IX ) .EQ. JL )  GOTO  46
   44 CONTINUE
      STOP 00044
   46 DDMAT = DDMAR( IX )
      RETURN
 
C*************************************************
C       NO  MATRIX  ELEMENT  ON  FILE - 10
C*************************************************
50    DDMAT =  DDME( JL, JR, K )
      RETURN
 
      END
      FUNCTION   D D M E  ( JL, JR, K )
      IMPLICIT  REAL*8  (A-H,O-Z )
      PARAMETER   (ICFP1=2907)
      COMMON / SPSTCM / ISTBS(300),NBD(300),LD(300),NSEN(300),NDL(300)
     &                 ,NBBAS(16),ISTBS2(300),LAD(300),LADBS(16)
      COMMON / CFP1CM / IDAU(ICFP1),L1(ICFP1),CFP1(ICFP1)
      COMMON / SR / SR0,SR(500)
C /--------------------------------------------------------------------/
C / ***   ( JL  !!  ( (D+) * (D-) ) (K)  !!  JR )                  *** /
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
      RETURN
      END















C     SUBROUTINE     N P B S E T
C     IMPLICIT  REAL*8(A-H,O-Z)
C     PARAMETER  (NM=72)
C     COMMON / OUTCM  / ICM(5)
C     COMMON / BCODCM / Q(2*NM*NM)
C
C     DO 210  I = 1, 5
C 210 ICM( I ) = 0
C     DO 230  I = 1, 2704
C 230 Q( I ) = 0
C     CALL   B C O D I N  ( 52 )
C     CALL   B C O D I N
C     RETURN
C     E N D














      SUBROUTINE  C O M I N 1  ( MAXND )
      IMPLICIT  REAL*8(A-H,O-Z)
      PARAMETER   ( ICFP1=2907, ICFP2=6371 )
      COMMON / SPSTCM /  ISTBS(300),NBD(300),LD(300),NSEN(300),NDL(300)
     1                  ,NBBAS(16),ISTBS2(300),LAD(300),LADBS(16)
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
      CALL   F C L O S E ( 11 )
      DO 203  J = 1, IL
  203 LIM(J) = IABS( LIM(J) )
      DO 204  J = 1, IL
  204 L2(J) = IABS( L2(J) )
      RETURN
  100 WRITE(6,680)
      ICMW = 3
      RETURN
600   FORMAT(/ T6,'# CFP FILE (#11) READ-IN   REQUIRED MAX(ND) =',I2
     &   /T16, '/ MAX(ND) ON FILE =',I2,' (IF FORMER > LATTER,',
     &   ' LATTER BE TAKEN)'/)
  680 FORMAT(//T7,'----->   ABNORMAL  TERMINATION  IN  READING',
     +  '  "CFPDK" .   ICMW  WAS  SET  3 .'//)
      END











      FUNCTION   D R A C  (JA1, JB1, LB1, LA1, JC1, LC1 )
C***********************************************************
C****    E N L A R G E D    V E R S I O N     **************
C***********************************************************
      IMPLICIT REAL*8(A-H,O-Z)
      PARAMETER  (NM=72)
      COMMON / BCODCM / Q(NM,NM), P(NM,NM)
C     DATA  IM  / ZFFFF0000 /
      DATA  IM  / -65536 /
C   ** Q(I,M) (I.GE.M) BINOMIAL COEF. C(I-1,M-1)
C   ** Q(M,I) (I.GE.M) Q(I,M)**(-1/2)
C   ** P(M,I) = Q(I,M)
      IE(K) = IAND(IM,K)
C** **
      JA = JA1
      JB = JB1
      LB = LB1
      LA = LA1
      JC = JC1
      LC = LC1
      GOTO  20
      ENTRY DSIXJ ( AJ, BJ, CJ, AL, BL, CL )
      IS = 0
      GOTO  7
      ENTRY DRACAH(AJ,BJ,BL,AL,CJ,CL)
      IFIXJ = AJ + BJ + BL + AL + 0.1
      IS = IAND( IFIXJ, 1 )
7     JA=2*AJ+0.1
      JB=2*BJ+0.1
      JC=2*CJ+0.1
      LA=2*AL+0.1
      LB=2*BL+0.1
      LC=2*CL+0.1
      GO TO 1
C** ** ARGUMENT 2*J'S  INTEGER TYPE
20    IS = IAND((JA+JB+LB+LA)/2, 1)
1     IF(Q(13,3).NE.66.0D0)  CALL   B C O D I N
      IF((IE(JA+JB-JC) + IE(JA+LB-LC) + IE(LA+LB-JC) + IE(LA+JB-LC)
     &  + IE(JB+JC-JA) + IE(JB+LC-LA) + IE(LB+LC-JA) + IE(LB+JC-LA)
     &  + IE(JC+JA-JB) + IE(JC+LA-LB) + IE(LC+LA-JB) + IE(LC+JA-LB))
     & .LT. 0 )  GOTO 70
C     IF( (JA+JB-JC) .LT. 0 .OR. (JA+LB-LC) .LT. 0 .OR.
C    &    (LA+LB-JC) .LT. 0 .OR. (LA+JB-LC) .LT. 0 .OR.
C    &    (JB+JC-JA) .LT. 0 .OR. (JB+LC-LA) .LT. 0 .OR.
C    &    (LB+LC-JA) .LT. 0 .OR. (LB+JC-LA) .LT. 0 .OR.
C    &    (JC+JA-JB) .LT. 0 .OR. (JC+LA-LB) .LT. 0 .OR.
C    &    (LC+LA-JB) .LT. 0 .OR. (LC+JA-LB) .LT. 0 )  GOTO  70
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
      IF(KMAX.GT.NM)GO TO 60
      X=0
      DO 10 K=KMIN,KMAX
C       X=-X+Q(K,MT+1)*Q(NA,K-MA)*Q(NB,K-MB)*Q(NC,K-MC)
        X=-X+Q(K,MT+1)*P(K-MA,NA)*P(K-MB,NB)*P(K-MC,NC)
10    CONTINUE
      DRAC  = X
     &   *Q(JA+2,MA+1)*Q(KA,JA+1)*Q(LA+2,MB+1)*Q(KB,LA+1)*Q(LA+2,MC+1)*
     &    Q(KC,LA+1)/(Q(JA+2,MT+1)*Q(NC,JA+1)*DFLOAT(LA+1))
      IF( IAND(KMAX+IS,1) .EQ. 1 )  DRAC = - DRAC
      RETURN
 
70    DRAC  = 0
      RETURN
   60 WRITE(6,602)  JA1, JB1, LB1, LA1, JC1, LC1
      GO TO 70
  602 FORMAT('DRAC/DRACAH   OUT OF RANGE  /  ',
     &       'W(', 3(I4,'/2,'), I4,'/2:', I4,'/2,', I4,'/2)')
      END











      SUBROUTINE  BCODIN
C***********************************************************
C****    E N L A R G E D    V E R S I O N     **************
C***********************************************************
      IMPLICIT REAL*8(A-H,O-Z)
      PARAMETER  (NM=72, NM2=NM*NM)
      COMMON /BCODCM/Q(NM2), P(NM,NM)
      DIMENSION      QX(NM,NM)
      EQUIVALENCE   (Q, QX)
 
C   **  COMMON /BCODCM/Q(NM,NM)
C   ** Q(M,I) (I.GE.M) BINOMIAL COEF. C(I-1,M-1)
C   ** Q(I,M) (I.GE.M) Q(M,I)**(-1/2)
C   ** P(I,M) = Q(M,I)
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
 
      DO  240  KA = 1, NM
        DO  242  KB = 1, NM
          P( KA, KB ) = QX( KB, KA )
242     CONTINUE
240   CONTINUE
 
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
