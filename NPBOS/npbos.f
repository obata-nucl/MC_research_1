C    CODED AND PROGRAMED BY T. OTSUKA.
C    THE USER SHOULD REFER TO THE "PROGRAM AND PROGRAMMER'S NAME".
C**************************************************
C ****   VERSION   FOR   VECTOR - PROCESSOR   ****
C**************************************************
C    Revised in May 1994 for FACOM/UXP in Kansai University.
C    Revised in Nov. 1993 for SUN4 of rknccs1 of RIKEN.
C    Revised in May 1992 for FACOM/UXP in RIKEN.
C    Revised in Apr. 1998 (KLMAX=120 <- KLMAX=96)
C    Revised in May. 2000 (DATE -> DATE_AND_TIME)
C    Revised in Oct. 2000 to fix imcompatible BCODIN argement
C           and common block lengths.
C    Revised in Apr. 2002 (NEST,NEMAX=20->50, KLMAX=150->600)
C    Revised in Dec. 2002 (SNSN, SNDN, DNDN, SPSP, ...  added)
      PROGRAM  NPBOS
      IMPLICIT  REAL*8(A-H,O-Z)
      PARAMETER   ( IHSTR=80000, IVSTR=6000 )
!      PARAMETER   ( IHSTR=500000, IVSTR=100000 )
      COMMON / LM / MAXDIM, NMAT, MAXST, IFL, NDRAC, LVEC
      COMMON / HMLCM / H(IHSTR), JIND(IHSTR)
 
      DIMENSION  WAV (IVSTR), VEC (IVSTR), IHBASE(IVSTR),
     &           ISTN(IVSTR), ISTP(IVSTR)
 
C
C File treatment modified so that the files are defined within the
C program.                                     (May 1992, N. Yoshida)
      open(3,STATUS='SCRATCH',ACCESS='SEQUENTIAL',FORM='UNFORMATTED')
C      OPEN(5,FILE='npbosin.dat',STATUS='OLD',ACCESS='SEQUENTIAL',
C    &     FORM='FORMATTED')
      open(6,FILE='out0.dat',STATUS='UNKNOWN',ACCESS='SEQUENTIAL',
     &     FORM='FORMATTED')
      open(8,FILE='out1.dat',STATUS='UNKNOWN',ACCESS='SEQUENTIAL',
     &     FORM='FORMATTED')
      open(9,file='racfl.dat',
     &     STATUS='OLD',ACCESS='SEQUENTIAL',FORM='UNFORMATTED')
      open(10,file='ddmefl.dat',
     &     STATUS='OLD',ACCESS='SEQUENTIAL',FORM='UNFORMATTED')
      open(11,file='cfp2.dat',
     &     STATUS='OLD',ACCESS='SEQUENTIAL',FORM='UNFORMATTED')
      open(13,STATUS='SCRATCH',ACCESS='SEQUENTIAL',FORM='UNFORMATTED')
      open(14,STATUS='SCRATCH',ACCESS='SEQUENTIAL',FORM='UNFORMATTED')
      open(20,FILE='npbosvec.dat',ACCESS='SEQUENTIAL',
     &     FORM='UNFORMATTED')
c      open(20,FILE='npbosvec.dat',STATUS='UNKNOWN',ACCESS='SEQUENTIAL',
c     &     FORM='UNFORMATTED')
!      OPEN(21,FILE='wf.dat',FORM='FORMATTED',ACCESS='SEQUENTIAL',
!     &     STATUS='UNKNOWN')
C
      MAXDIM = IHSTR
      MAXST  = IVSTR

      C A L L   N P L A N  ( WAV, VEC, IHBASE, ISTN, ISTP )

      END

      subroutine   N P L A N  ( WAV, VEC, IHBASE, ISTN, ISTP )
      IMPLICIT  REAL*8(A-H,O-Z)
C *******************************************************************
C ***     THIS  SUBPROGRAM  IS  THE  ACTUAL  MAIN  PROGRAM .
C **
C ***     THIS  SET  OF  PROGRAMS  FOR  DIAGONALIZATION  OF  THE
C ***     NEUTRON-PROTON  BOSON  HAMILTONIAN  IS  CALLED  'N P B O S' .
C **
C :::     --->  IF  ANY  PROBLEM  IS  FOUND,  PLEASE  WRITE  TO
C :::
C :::               TAKAHARU  OTSUKA
C :::               PHYSICS DEPARTMENT
C :::               JAPAN ATOMIC ENERGY RESEARCH INSTITUTE
C :::               TOKAI, IBARAKI 319-11
C :::               JAPAN
C :::
C *******************************************************************
C *****
C **   FILE -  3  :  Hamiltonian  matrix  storage
C **          20  :  eigenvectors  storage
C **          13  :  TEMPORARY  USE  (LANCZOS  BASIS  STORAGE)
C **          14  :  TEMPORARY  USE  (EIGENVECTOR  STORAGE)
C *****
      PARAMETER   ( NEST=50, KLMAX=600 )
      PARAMETER  (NM=72)
      CHARACTER*1  ITYPE
      CHARACTER*40 IVER10
      CHARACTER*10  DAY,YMD,HMS,STT
      CHARACTER*8  ITM
      CHARACTER*1  ITM1
      CHARACTER*3  MASS
      CHARACTER*3  NUCL
      CHARACTER    IVER*4, ITITLE*4, DAYV*8, TMV*8
      PARAMETER   ( IHSTR=80000, IVSTR=6000 )
!      PARAMETER   ( IHSTR=500000, IVSTR=100000 )
      COMMON / HMLCM / H(IHSTR), JIND(IHSTR)
      COMMON / NPST / NDN(200),NDP(200),NPBS(200),
     1                NDT(20),NDTBS(20),NQBS(-3:20),NST
      COMMON / SR / SR(0:500)
C     COMMON /BCODCM / Q(2704)
      COMMON /BCODCM / Q(2*NM*NM)
!      COMMON /BCODCM / Q(10816)
      COMMON / NBCNT / NDUPT, NDUPN, NDUPP, L, NBSN, NBSP
      COMMON / H / EFIX,
     1             ESPN, C0N, C2N, C4N, VYN, VWN, VAN, VBN, VCN,
     2             ESPP, C0P, C2P, C4P, VYP, VWP, VAP, VBP, VCP,
     3             VAV, VXV, VVX, VXX0, VXX(4), VVV,
     4             SNSP, DNSP, SNDP, DNDP, VSSN, VSSP,
     &             SNSN,SNDN,DNDN,SPSP,SPDP,DPDP,ESN,ESP
      COMMON / LAN   / NEIG, NVEC, EPS, LFILE, NCUT, BEFIX, ISYM,
     &                 IFILE, TRVASM
      COMMON / O     / ENERGY(KLMAX), JSTAT(KLMAX), FSTAT(KLMAX),
     &                 KLEVEL, KLVL1
      COMMON / VERF  / IVER(10), ITITLE(2), DAYV(2), TMV
      COMMON / OUTCM / ICMW, IDBG, NPSTW, MATW, IWCF
      COMMON / LM    / MAXDIM, NMAT, MAXST, IFL, NDRAC, LVEC
      COMMON / EXPCM  / NEXP, NEXP1, JEXP(KLMAX), IPEXP(KLMAX),
     &                  KCEXP(KLMAX), EXEXP(KLMAX)
      PARAMETER   (IDDM=7594)
      COMMON / DDMECM /  MAXDD,NDBST,NDBST1,NMEDD,
     &                   IBSDD(300,3),ILDDM(IDDM),DDMAR(IDDM)
      PARAMETER   ( NEMAX=50 )
      COMMON / QQEXCM / CHN, CHP, EXQ (NEMAX,NEMAX,3),
     &                  EXN (NEMAX,NEMAX,2), EXF (NEMAX,NEMAX)
 
      DIMENSION   WAV(*), VEC(*), IHBASE(*), ISTN(*), ISTP(*)
 
      DIMENSION   LAUTO(32), NEIGA(32), NDUPTA(32)
      DIMENSION   ITM1 (8)
      EQUIVALENCE  ( DAY, DAYV ), (ITM, TMV ), ( IVER(1), IVER10 )
      EQUIVALENCE  ( ITM, ITM1 )
      EQUIVALENCE  ( ITITLE(1), MASS ), ( ITITLE(2), NUCL )
      NAMELIST / N / ICMW, IDBG, NPSTW, MATW, NVEC1,
     1               NCUT, IWCF, IWDDME, ISYM, TRVASM, NDRAC,
     2               LAUTO, NEIGA, NDUPTA, IEX, eps
C
c     CALL   SECOND( CTM )
      IVER10 = 'nov-89 (MAR-85 LA)/ TKYVAX / TAKA OTSUKA'
      IFILE  = 14
      MAXND  = 12
      NDRAC  = 10
!      MAXND  = 15
!      NDRAC  = 15
      IEX    = 1
      IWDDME = 1
      ISYM   = 0
      TRVASM = 0.2
      NEIG1  = 0
      NVEC1  = 0
      NCUT   = 30
      NT0    = 0
      C A L L     N P B S E T
      EPS = 1.D-8
      READ(5, N )
      WRITE(6, N )
C     CALL  DATE ( DAY )
c     CALL DATE_AND_TIME(YMD,HMS,STT,I)
c     DAY=YMD(7:8)//'.'//YMD(5:6)//'.'//YMD(1:4)
c     CALL  ATIME ( ITM )
      DAY='          '
      ITM='        '
      ITM1(3) = ':'
      ITM1(6) = ':'
      WRITE(6,610) DAY, ITM
      CALL  C O M I N 1   ( MAXND )
      IF( ICMW .LE. 0 )  GO  TO  7
c *** CALL  C O M O U T   ( MAXND, ICMW )
      IF( NDRAC .LT. 0 )  NDRAC = 0
 
7     SR( 0 ) = 0
      DO 5  I = 1, 500
      FI = I
5     SR( I ) = DSQRT( FI )
 
      CALL  D D M E I N  ( IWDDME )
 
      NEIG = NEIG1
      NVEC = NVEC1

C      inputting the data
3     READ(5,500)  MASS, NUCL
C      mass number and element name of nucleus
      WRITE(6,700) MASS, NUCL
      WRITE(8,700) MASS, NUCL
 
      READ(5,503)  NBSNT, NBSPT
C      numbers of neutron and proton bosons
C-----------------------------------------
C  ****    FOR  ISU3=1  IN  FSUB    ****
C-----------------------------------------
      NBSN = NBSNT
      NBSP = NBSPT
C-----------------------------------------
 
      READ(5,503) NEXP
      IF( NEXP .NE. 0 )  THEN
        WRITE(6,662)  NEXP
        DO 992  J = 1, NEXP
          READ(5,560)  JEXP(J), IPEXP(J), KCEXP(J), EXEXP(J)
          WRITE(6,660)  JEXP(J), IPEXP(J), KCEXP(J), EXEXP(J)
992     CONTINUE
      ENDIF
C***************************
      CALL   F S U B
C***************************
      WRITE(8,640)  DAY, ITM
 
4     READ(5,501)  ITYPE
      KLEVEL = 0
      IMODE = -1
      IF( ITYPE. EQ. ' ' )  IMODE = 0
      IF( ITYPE. EQ. 'P' )  IMODE = 1
      IF( ITYPE. EQ. 'N' )  IMODE = 2
      IF( IMODE .LE. -1 )  STOP  'NORMAL TERMINATION'
 
      GOTO ( 10, 11, 12, 13 ), IMODE + 2
11      NBSN = NBSNT
        NBSP = NBSPT
        GOTO  10
 
12      NBSN = NBSNT
        NBSP = NBSPT - 1
        GOTO  10
 
13      NBSN = NBSNT - 1
        NBSP = NBSPT
        GOTO  10
 
C----------------------------------------------------------
 
10    LFILE = 20 + IMODE
      DO 9  I = 1, 96
9     FSTAT( I ) = -10
 
      IAST = 0
 
1     IAST = IAST + 1
      L = LAUTO( IAST )
C----------------------------------------------------------
      IF( L .LT. 0 )  THEN
        IF( NEIG .GT. 0 )  CALL   B O S O U T  ( VEC(  1), VEC(201),
     &                                           VEC(401), VEC(601),
     &                                           VEC(801) )
 
        GOTO  4
      ENDIF
C---------------------------------------------------------
      NE = NEIGA(IAST)
      IF( NE .LE. 0 )  NE = NEIG1
      NEIG1 = NE
      NVEC1 = NE
      NEIG = NEIG1
      NVEC = NVEC1
 
      IF( NDUPTA(IAST) .GT. 0 )  NT0 = NDUPTA(IAST)
      IF( NT0 .GT. 0 )  THEN
        NT = NT0
      ELSE
        NT = NBSN + NBSP
      ENDIF
      NDUPTX = NT
C ************************** IF( NEIG .GT.  5  )  NEIG = 5
      IF( NVEC .GT.  NEST )  NVEC =  NEST
      IF( NVEC .LT. -NEST )  NVEC = -NEST
C*********************************
      CALL   N P B S T   ( NDUPTX, MAXND, ISTN, ISTP )
C*********************************
      IF( NST .EQ. 0 )  GO  TO  1
C****
      IF( NEIG .LE. 0 .AND. IDBG .EQ. 0 )  GO  TO  1
C****
c     CALL    SECOND  ( CTM1 )
C******************************
      CALL    B S Y S M E    ( WAV, IHBASE, ISTN, ISTP )
C******************************
c     CALL    SECOND  ( CTM2 )
C***
      IF(IWCF.LE.0) GOTO 102
      CTM= ( CTM2 - CTM1 )
      WRITE(6,630)  CTM
C****
      IF( IWCF .GE. 1 )  WRITE(6,621)
C****
  102 IF(NEIG.LE.0) GOTO 1
C******************************
      CALL    E I G L A N    ( WAV, VEC, IMODE, IHBASE, ISTN, ISTP )
C******************************
      IF(IWCF.LE.0) GOTO 103
c     CALL    SECOND  ( CTM3 )
      CTM= CTM3 - CTM2
      WRITE(6,631)  CTM
C***********************************************************************
103   IF( IEX .NE. 0 )  THEN
        IVEC = LVEC
        IF( LVEC .GT. ((3*MAXDIM)/(NST*2)) )  IVEC = (3*MAXDIM)/(NST*2)
 
        CALL    Q Q E X   ( H, NST, IVEC, IEX, ISTN, ISTP )
 
      ENDIF
 
      GOTO  1
C*****                     *****
C*****     F O R M A T     *****
C*****                     *****
500   FORMAT( A3, 1X, A2 )
501   FORMAT(A1,I4,15I5)
502   FORMAT( 8F10.7 )
503   FORMAT( 6I5 )
504   FORMAT( F10.7, I5, F10.7 )
560   FORMAT( I5,1X,A1,1X,I1,F12.5 )
  610 FORMAT('1' /// T5,'***********   N P B O S',
     &   '   C A L C U L A T I O N    (VERSION R)   ***********'
     &  /T5,'*', T80,'*'
     &  /T5,'*', T20,'VERSION   MAR  29  1985  //',
     &    '   LOS ALAMOS-TOKAI', T80,'*'
     &  /T5,'*', T80,'*'
     &  /T5,'*', T20,'DATE : ',A10, T40,'TIME : ',A8,T80,'*'
     &  /T5,'*', T80,'*'
     &  /T5,76('*') //)
621   FORMAT( /T3,'*****   DIAGONALIZATION  OF  HAMILTONIAN   *****' /)
  630 FORMAT(//T12,'**  C-TIME  IN  BSYSME  :',F7.1,'  (SEC)'/)
  631 FORMAT(/ T12,'**  C-TIME  IN  EIGLAN  :',F7.1,'  (SEC)')
640   FORMAT( /T3,'DATE : ',A10, T24,'TIME : ',A8 )
  600 FORMAT( '1' / )
  660 FORMAT( T14,I2,1X,A1,2X,'  (',I1,')  ',F12.5)
  662 FORMAT(/T7,'**  EXPERIMENTAL  SPECTRUM   (# OF DATA =',I4,' )'
     &     /  T14,'SPIN',T29,'EXP. ENERGY (MEV)' )
700   FORMAT( //T5, '*----------*'
     &        / T5, '*  ', A3, T16,'*'
     &        / T5,'*', T12,A2,T16,'*'
     &        / T5, '*----------*' )
      E N D













 
      subroutine    F S U B
C***********************************************************************
C*****  ELEMENT   FSUB   (VERSION  #R02)   *****  22  JUL  1984    *****
C***********************************************************************
      IMPLICIT  REAL*8(A-H,O-Z)
      COMMON / NBCNT / NDUPT, NDUPN, NDUPP,LL, NBSN, NBSP
      COMMON / H / EFIX,
     1             ESPN, C0N, C2N, C4N, VYN, VWN, VAN, VBN, VCN,
     2             ESPP, C0P, C2P, C4P, VYP, VWP, VAP, VBP, VCP,
     3             VAV, VXV, VVX, VXX(0:4), VVV,
     5             SNSP, DNSP, SNDP, DNDP, VSSN, VSSP,
     &             SNSN,SNDN,DNDN,SPSP,SPDP,DPDP,ESN,ESP
      COMMON / EQQ   / FEQ(2),
     1     QCN, QSPN, QC0N, QC2N, QC4N, QVYN, QVWN, QVAN, QVBN, QVCN,
     2     QCP, QSPP, QC0P, QC2P, QC4P, QVYP, QVWP, QVAP, QVBP, QVCP
      COMMON/ PAR / RKAP, EDN, EDP, CLN, CLP, RLNN, RLPP, RLNP, ED,
     &              GK(5,2), RKAP0, RKAP2, DSNDDP, DSPDDN, QN, QP,
     &              RKNN, RKPP, PPN, PPP, RMAJ, RMAJ1, RMAJ3, RMAJ2
      PARAMETER   ( NEMAX=50 )
      COMMON / QQEXCM / CHN, CHP, EXQ (NEMAX,NEMAX,3),
     &                  EXN (NEMAX,NEMAX,2), EXF (NEMAX,NEMAX)
      COMMON/ SR     / SR0, SR(500)
      DIMENSION    GNP(0:4)
      DIMENSION    FDDUM(33), FDUM(1), FQDUM(20)
      EQUIVALENCE (RKAP, FDDUM(1)), (EFIX, FDUM(1))
      EQUIVALENCE  (FEQ(1), FQDUM(1))
      NAMELIST/ INPT / RKAP, RKAP0, RKAP2, ED, EDN, EDP, FEQ, QN, QP,
     +                 EFIX, CHN, CHP, GNP, VXX, HEX,
     +                 C0N, C2N, C4N, C0P, C2P, C4P,
     &                 PPN, PPP, RMAJ, RMAJ1, RMAJ3, RMAJ2,
     +                 CLN, CLP, RKNN, RKPP, RLNN, RLPP, RLNP, ISU3,
     &                 SNSP, DNSP, SNDP, DNDP, VWN, VYN, VWP, VYP,
     &                 SNSN,SNDN,DNDN,SPSP,SPDP,DPDP,ESN,ESP
C **********************************************************************
      DO 202 I = 1, 42
  202 FDUM ( I ) = 0
      DO 201 I = 1, 22
  201 FQDUM ( I ) = 0
      DO 205 I = 1, 33
  205 FDDUM ( I ) = 0
      CHN = 0
      CHP = 0
      ISU3 = 0
      DO 220  I = 0, 4
        GNP( I ) = 0
  220 CONTINUE
 
      READ(5,INPT)
      VXX(4) = VXX(4) + HEX
      WRITE(6,610) RKAP, HEX, EFIX, CHN, CHP, RLNN, RLPP, RLNP, ED, EDN,
     &             EDP, RMAJ, RMAJ2, RMAJ1, RMAJ3, RKNN, RKPP, ISU3
 
      IF( FEQ(1).NE. 0 .OR.  FEQ(2).NE. 0 )  WRITE(6,611) FEQ
      IF( RKAP0 .NE. 0 .OR. RKAP2 .NE. 0 )  WRITE(6,616) RKAP0, RKAP2
      IF( CLN .NE. 0 .OR. CLP .NE. 0 .OR.
     &    GNP(0) .NE. 0 .OR. GNP(2) .NE. 0 .OR. GNP(4) .NE. 0 .OR.
     &    GNP(1) .NE. 0 .OR. GNP(3) .NE. 0 )
     &          WRITE(6,612) CLN,CLP,GNP(0),GNP(2),GNP(4),GNP(1),GNP(3)
      IF( SNSP .NE. 0 .OR. DNSP .NE. 0 .OR.
     &    SNDP .NE. 0 .OR. DNDP .NE. 0 .OR.
     &    DSNDDP.NE. 0 .OR. DSPDDN.NE. 0 )
     &         WRITE(6,614) SNSP, DNSP, SNDP, DNDP, DSNDDP, DSPDDN
      IF(ESN.NE.0.OR.ESP.NE.0) WRITE(6,619) ESN,ESP
      IF(SNSN.NE.0.OR.SNDN.NE.0.OR.DNDN.NE.0)
     &      WRITE(6,615) SNSN,SNDN,DNDN
      IF(SPSP.NE.0.OR.SPDP.NE.0.OR.DPDP.NE.0)
     &      WRITE(6,617) SPSP,SPDP,DPDP
      IF( C0N .NE. 0 .OR. C2N .NE. 0 .OR. C4N .NE. 0 .OR.
     &    C0P .NE. 0 .OR. C2P .NE. 0 .OR. C4P .NE. 0 )
     &         WRITE(6,618) C0N, C2N, C4N, C0P, C2P,  C4P
      IF( VWN .NE. 0 .OR. VYN .NE. 0 .OR.
     &    VWP .NE. 0 .OR. VYP .NE. 0 )
     &         WRITE(6,622) VWN, VYN, VWP, VYP
      WRITE(8,610) RKAP, HEX, EFIX, CHN, CHP, RLNN, RLPP, RLNP, ED, EDN,
     &             EDP, RMAJ, RMAJ2, RMAJ1, RMAJ3, RKNN, RKPP, ISU3
      IF( FEQ(1).NE. 0 .OR.  FEQ(2).NE. 0 )  WRITE(8,611) FEQ
      IF( RKAP0 .NE. 0 .OR. RKAP2 .NE. 0 )  WRITE(8,616) RKAP0, RKAP2
      IF( CLN .NE. 0 .OR. CLP .NE. 0 .OR.
     &    GNP(0) .NE. 0 .OR. GNP(2) .NE. 0 .OR. GNP(4) .NE. 0 .OR.
     &    GNP(1) .NE. 0 .OR. GNP(3) .NE. 0 )
     &          WRITE(8,612) CLN,CLP,GNP(0),GNP(2),GNP(4),GNP(1),GNP(3)
      IF( SNSP .NE. 0 .OR. DNSP .NE. 0 .OR.
     &    SNDP .NE. 0 .OR. DNDP .NE. 0 .OR.
     &    DSNDDP.NE. 0 .OR. DSPDDN.NE. 0 )
     &         WRITE(8,614) SNSP, DNSP, SNDP, DNDP, DSNDDP, DSPDDN
      IF(ESN.NE.0.OR.ESP.NE.0) WRITE(8,619) ESN,ESP
      IF(SNSN.NE.0.OR.SNDN.NE.0.OR.DNDN.NE.0)
     &      WRITE(8,615) SNSN,SNDN,DNDN
      IF(SPSP.NE.0.OR.SPDP.NE.0.OR.DPDP.NE.0)
     &      WRITE(8,617) SPSP,SPDP,DPDP
      IF( C0N.NE. 0 .OR. C2N.NE. 0 .OR. C4N.NE. 0 .OR.
     &    C0P.NE. 0 .OR. C2P.NE. 0 .OR. C4P.NE. 0 )
     &         WRITE(8,618) C0N, C2N, C4N, C0P, C2P, C4P
      IF( VWN .NE. 0 .OR. VYN .NE. 0 .OR.
     &    VWP .NE. 0 .OR. VYP .NE. 0 )
     &         WRITE(8,622) VWN, VYN, VWP, VYP
      VVV = RKAP + RKAP2
      VAV = ( RKAP + RKAP0 )
      RMAJ2 = RMAJ2 + RMAJ
      RMAJ1 = RMAJ1 + RMAJ
      RMAJ3 = RMAJ3 + RMAJ
      VAV   = VAV   - RMAJ2 / 2
 
        IF( ISU3 .EQ. 1 )  THEN
          CHN = - SR(7) / 2
          CHP = CHN
        ENDIF
 
      VXV = RKAP * CHN + DSPDDN
      VVX = RKAP * CHP + DSNDDP
      ESPN =ED+ EDN
      ESPP =ED+ EDP
      SNDP = SNDP + RMAJ2 / 2
      DNSP = DNSP + RMAJ2 / 2
      VXX(1) = VXX(1) + RLNP * 10
      VXX(2) = VXX(2) + RKAP * CHN * CHP
      GNP(1) = GNP(1) + RMAJ1
      GNP(3) = GNP(3) + RMAJ3
      DO 300  K = 0, 4
        X = 0
        DO 310  L = 0, 4
          Y = GNP( L ) * ( 2*L + 1 )
     &      * DRAC( 4, 4, 4, 4, L*2, K*2 )
          IF( IAND( 1, L ) .EQ. 1 )  Y = - Y
          X = X + Y
310     CONTINUE
        VXX( K ) = VXX( K ) + X
300   CONTINUE
C***********************************************************************
C*******     INTERACTION    BETWEEN    L I K E    B O S O N S     ******
C***********************************************************************
      C0N = -12 * RLNN + CLN + C0N
      C2N = - 6 * RLNN + CLN + C2N
      C4N =   8 * RLNN + CLN + C4N
      C0P = -12 * RLPP + CLP + C0P
      C2P = - 6 * RLPP + CLP + C2P
      C4P =   8 * RLPP + CLP + C4P
 
      IF( ISU3 .EQ. 1 )  THEN
        RKNN = RKAP / 2
        CHN = - SR(7) / 2
      ENDIF
      ESPN = ESPN + RKNN * ( (2*(NBSN-3)) + (CHN**2) ) + 6 * RLNN
      C0N = C0N + RKNN * ( -4 + 10*CHN*CHN*DRAC(4,4,4,4,4,0) )
      C2N = C2N + RKNN * ( -4 + 10*CHN*CHN*DRAC(4,4,4,4,4,4) )
      C4N = C4N + RKNN * ( -4 + 10*CHN*CHN*DRAC(4,4,4,4,4,8) )
      VYN = VYN + 2 * SR(2) * RKNN * CHN
      VWN = VWN + 2 * SR(5) * RKNN
      EFIX = EFIX + RKNN * 5 * NBSN
 
      IF( ISU3 .EQ. 1 )  THEN
        RKPP = RKAP / 2
        CHP = - SR(7) / 2
      ENDIF
      ESPP = ESPP + RKPP * ( (2*(NBSP-3)) + (CHP**2) ) + 6 * RLPP
      C0P = C0P + RKPP * ( -4 + 10*CHP*CHP*DRAC(4,4,4,4,4,0) )
      C2P = C2P + RKPP * ( -4 + 10*CHP*CHP*DRAC(4,4,4,4,4,4) )
      C4P = C4P + RKPP * ( -4 + 10*CHP*CHP*DRAC(4,4,4,4,4,8) )
      VYP = VYP + 2 * SR(2) * RKPP * CHP
      VWP = VWP + 2 * SR(5) * RKPP
      EFIX = EFIX + RKPP * 5 * NBSP
C**********************************************************************
      VAN = VAN + ( 3 * C4N + 4 * C2N ) / 7
      VCN = VCN + ( C4N - C2N ) / 14
      VBN = VBN + ( C0N - VAN + 12 * VCN ) / 10
      VAP = VAP + ( 3 * C4P + 4 * C2P ) / 7
      VCP = VCP + ( C4P - C2P ) / 14
      VBP = VBP + ( C0P - VAP + 12 * VCP ) / 10
C***********************************************************************
C****   PARAMETERS  FOR  ED*(Q*Q)  INTERACTION    **********************
C***********************************************************************
      QCN = ( 5 * NBSN ) * FEQ(2)
      QCP = ( 5 * NBSP ) * FEQ(1)
      QSPN = ( ( 2 * NBSN - 6 ) + CHN*CHN ) * FEQ(2)
      QSPP = ( ( 2 * NBSP - 6 ) + CHP*CHP ) * FEQ(1)
      QC0N = ( -4 + 10*CHN*CHN*DRAC(4,4,4,4,4,0) ) * FEQ(2)
      QC2N = ( -4 + 10*CHN*CHN*DRAC(4,4,4,4,4,4) ) * FEQ(2)
      QC4N = ( -4 + 10*CHN*CHN*DRAC(4,4,4,4,4,8) ) * FEQ(2)
      QC0P = ( -4 + 10*CHP*CHP*DRAC(4,4,4,4,4,0) ) * FEQ(1)
      QC2P = ( -4 + 10*CHP*CHP*DRAC(4,4,4,4,4,8) ) * FEQ(1)
      QC4P = ( -4 + 10*CHP*CHP*DRAC(4,4,4,4,4,8) ) * FEQ(1)
      QVWN = 2 * SR(5) * FEQ(2)
      QVWP = 2 * SR(5) * FEQ(1)
      QVYN = 2 * SR(2) * CHN * FEQ(2)
      QVYP = 2 * SR(2) * CHP * FEQ(1)
      QVAN = ( 3*QC4N + 4*QC2N ) / 7
      QVCN = ( QC4N - QC2N ) / 14
      QVBN = ( QC0N - QVAN + 12*QVCN ) / 10
      QVAP = ( 3*QC4P + 4*QC2P ) / 7
      QVCP = ( QC4P - QC2P ) / 14
      QVBP = ( QC0P - QVAP + 12*QVCP ) / 10
      RETURN
C*****                                  *****
C***                 F  O  R  M  A  T     ***
C*****                                  *****
  610 format(/T3,'***  parameters  in  the  Hamiltonian  ***'//
     & T6,'RKAP :',F7.3, T24,'HEX  :',F7.3, T42,'EFIX :',F7.3/
     & T6,'CHN  :',F7.3, T24,'CHP  :',F7.3 /
     & T6,'RLNN :',F7.3, T24,'RLPP :',F7.3, T42,'RLNP :',F7.3/
     & T6,'ED   :',F7.3, T24,'EDN  :',F7.3, T42,'EDP  :',F7.3/
     & T6,'MAJ  :',F7.3, T24,'S-D  :',F7.3, T42,'J=1 :',F6.3,
     & T58,'J=3 :',F6.3/
     & T6,'RKNN :',F7.3, T24,'RKPP :',F7.3,T42,'ISU3(1/0) :',I2/)
  611 FORMAT(T6,'FEQ(N) :',F7.3, T24,'FEQ(P) :',F7.3)
  612 FORMAT( T6,'CLN  :',F7.3, T24,'CLP  :',F7.3/
     & T6,'GNP0 :',F7.3, T24,'GNP2 :',F7.3,T42,'GNP4 :',F7.3/
     & T6,'GNP1 :',F7.3, T24,'GNP3 :',F7.3)
  614 FORMAT( T6,'SNSP :',F7.3, T24,'DNSP :',F7.3,
     & T42,'SNDP :',F7.3,T60,'DNDP1:',F7.3 /
     & T6,'DNDP2 :',F7.3, T24,'DNDP3 :',F7.3,
     & T42,'OMGN  :',F7.3,T60, 'OMGP  :',F7.3 /
     & T6,'DSNDDP:',F7.3, T24, 'DSPDDN:',F7.3)
  615 FORMAT( T6,'SNSN :',F7.3, T24,'SNDN :',F7.3,
     & T42,'DNDN :',F7.3)
616   FORMAT( T6,'RKAP0:',F7.3, T24,'RKAP2:',F7.3)
  617 FORMAT( T6,'SPSP :',F7.3, T24,'SPDP :',F7.3,
     & T42,'DPDP :',F7.3)
  618 FORMAT( T6,'C0N  :',F7.3, T24,'C2N  :',F7.3,T42,'C4N  :',F7.3/
     & T6,'C0P  :',F7.3, T24,'C2P  :',F7.3,T42,'C4P  :',F7.3 )
  619 FORMAT( T6,'ESN  :',F7.3, T24,'ESP  :',F7.3)
  620 FORMAT( T6,'ESFB(N) :',F7.3, T24,'ESFB(P) :',F7.3/
     & T6,'RKF  :',F7.3, T24,'PSN  :',F7.3,T42,'PSP  :',F7.3/
     & T6,'ETN  :',F7.3, T24,'ETP  :',F7.3/
     & T6,'DF (L):',5(F7.3,4X)/
     & T6,'FD (L):',5(F7.3,4X)/
     & T6,'FDX(L):',5(F7.3,4X)/
     & /)
622   FORMAT( T6, 'VWN  :',F7.3, T24,'VYN  :',F7.3,
     &        T42,'VWP  :',F7.3, T60,'VYP  :',F7.3 )
      END








      SUBROUTINE   B O S O U T  ( ELEVEL, JLEVEL, KCOUNT, FLEVEL,
     &                            EXPEN )
      IMPLICIT  REAL*8(A-H,O-Z)
      PARAMETER  ( KLMAX= 600 )
      REAL*4        CTM
      character*40  IVER10
      CHARACTER*8   ITM
      CHARACTER*1   ILEVEL, ITM1
C     CHARACTER     IVER*4, ITITLE*3, DAYV*5, TMV*8
      CHARACTER     IVER*4, ITITLE*4, DAYV*8, TMV*8
      COMMON / NBCNT / NDUPT, NDUPN, NDUPP, L, NNB2, NPB2
      COMMON / LAN   / NEIG, NVEC, EPS, LFILE, NCUT, BEFIX, ISYM,
     &                 IFILE, TRVASM
      COMMON / O   / ENERGY(KLMAX), JSTAT(KLMAX), FSTAT(KLMAX),
     &               KLEVEL, KLVL1
      COMMON / EXPCM / NEXP, NEXP1, JEXP(KLMAX), IPEXP(KLMAX),
     &                 KCEXP(KLMAX), EXEXP(KLMAX)
      COMMON / VERF  / IVER(10), ITITLE(2), DAYV(2), TMV
      DIMENSION   ITM1(8)
      DIMENSION   ELEVEL(KLMAX), JLEVEL(KLMAX), KCOUNT(KLMAX), 
     &            FLEVEL(KLMAX), EXPEN(KLMAX)
      EQUIVALENCE  ( IVER, IVER10 ), (ITM,ITM1)
 
      FMAX = NNB2 + NPB2
      FMAX = FMAX / 2
      FMAX2 = FMAX * ( FMAX + 1 )
      DO  80  I = 1 , KLEVEL
   80 EXPEN(I) = -1000.
      DO  20  I = 1 , KLEVEL
      EN1 = 1000.
      DO  10  J = 1 , KLEVEL
      ENL = ENERGY(J)
      IF ( ENL .GT. EN1 )         GO  TO   10
      JJ = J
      EN1 = ENL
   10 CONTINUE
      ELEVEL(I) = EN1
      ENERGY(JJ) = 1000.
      JLEVEL(I) = JSTAT(JJ)
      FLEVEL(I) = FSTAT(JJ)
C *** KCONV(I) = ICONV(JJ)
   20 CONTINUE
      BE = - ELEVEL(1)
C *************************************************
      DO 70  I = 1, KLEVEL
      JJ = 0
      DO 72  J = 1, I
      IF( JLEVEL(J) .EQ. JLEVEL(I) )
     &     JJ = JJ + 1
   72 CONTINUE
      KCOUNT(I) = JJ
      IF( NEXP .EQ. 0 )  GOTO  70
      DO 74 K = 1, NEXP
      IF( JEXP(K) .NE. JLEVEL(I)
     &    .OR. KCEXP(K) .NE. JJ )  GOTO  74
      EXPEN(I) = EXEXP(K)
      GOTO  70
   74 CONTINUE
   70 CONTINUE
C *************************************************
      DO  30  I = 1 , KLEVEL
c   30 ELEVEL(I) = ELEVEL(I) + EFIX
c modified on Aug.1st,2009
   30 ELEVEL(I) = ELEVEL(I) + BE
c     CALL    ATIME ( ITM )
c     CALL    SECOND( CTM )
      ICTM = CTM
      ITM1(3) = ':'
      ITM1(6) = ':'
c     WRITE(6,600)  ICTM,  ITM
c     WRITE(8,600)  ICTM,  ITM
      WRITE(6,115)
      WRITE(6,150)  IVER
      WRITE(6,140)  NNB2, NPB2, FMAX2, BE
      WRITE(8,150)  IVER
      WRITE(8,140)  NNB2, NPB2, FMAX2, BE
      DO 50  I = 1, KLEVEL
      ILEVEL = '+'
      EN1 = ELEVEL(I) / ELEVEL(2)
      IF( EXPEN(I) .LT. -10.0 )  GOTO  90
      IF( FLEVEL(I) .GE. 0 )  THEN
        FR = 100 * FLEVEL(I) / FMAX2
        WRITE(6,120)  I, JLEVEL(I), ILEVEL   , KCOUNT(I),
     &                ELEVEL(I), EXPEN(I), EN1, FR
        WRITE(8,120)  I, JLEVEL(I), ILEVEL   , KCOUNT(I),
     &                ELEVEL(I), EXPEN(I), EN1, FR
      ELSE
        WRITE(6,120)  I, JLEVEL(I), ILEVEL   , KCOUNT(I),
     &                ELEVEL(I), EXPEN(I), EN1
        WRITE(8,120)  I, JLEVEL(I), ILEVEL   , KCOUNT(I),
     &                ELEVEL(I), EXPEN(I), EN1
      ENDIF
      GOTO  50
90    IF( FLEVEL(I) .GE. 0 )  THEN
        FR = 100 * FLEVEL(I) / FMAX2
        WRITE(6,122)  I, JLEVEL(I), ILEVEL   , KCOUNT(I),
     &              ELEVEL(I), EN1, FR
        WRITE(8,122)  I, JLEVEL(I), ILEVEL   , KCOUNT(I),
     &              ELEVEL(I), EN1, FR
      ELSE
        WRITE(6,122)  I, JLEVEL(I), ILEVEL   , KCOUNT(I),
     &              ELEVEL(I), EN1
        WRITE(8,122)  I, JLEVEL(I), ILEVEL   , KCOUNT(I),
     &              ELEVEL(I), EN1
      ENDIF
50    CONTINUE
C
115   FORMAT(1H1)
120   FORMAT(/T3,I3, T10,I2, T14,A1, T17,'(',I2,')', T22,2F10.3,
     &        T46,F8.3, T60,F6.1 )
122   FORMAT(/T3,I3, T10,I2, T14,A1, T17,'(',I2,')', T22, F10.3,
     &        T46,F8.3, T60,F6.1 )
140   FORMAT(/T3,16('****')/T3,'***',5X,I2,
     1  ' - NEUTRON BOSONS', 9X, I2, ' - PROTON BOSONS',  T64,'***'
     2  /T3,'***', T20,'F*F (MAX) =', F7.2,               T64,'***'
     3  /T3,'***', T20,'BINDING ENERGY =', F9.4,' (MEV)', T64,'***'
     4  /T3,16('****')
     5  // T4,'NO.', T11,'J  P', T24,'EXC (MEV)  EXPERIMENT    RATIO',
     6     T59,'F*F/MAX (%)' / )
150   FORMAT(/T3,'NPBOS VERSION : ',10A4 )
c600  FORMAT(/T3,'CPU-TIME :', I8, ' (SEC)',8X,
c    &   'FINISHED AT ', A8 )
      RETURN
      END
      SUBROUTINE     N P B S T   ( NDUPTX, MAXND, ISTN, ISTP )
      IMPLICIT  REAL*8(A-H,O-Z)
      PARAMETER   (ICFP1=2907, ICFP2=6371)
      COMMON / SPSTCM /  ISTBS(300),NBD(300),LD(300),NSEN(300),NDL(300)
     1                  ,NBBAS(16),ISTBS2(300),LAD(300),LADBS(16)
      COMMON / CFP1CM / IDAU(ICFP1), L1(ICFP1), CFP1(ICFP1)
      COMMON / CFP2CM / IDAU2(ICFP2),L2(ICFP2),CFP2(ICFP2),LIM(ICFP2)
      COMMON / NPST / NDN(200),NDP(200),NPBS(200),
     1                NDT(20),NDTBS(20),NQBS(-3:20),NSTF
      COMMON / NBCNT / NDUPT, NDUPN, NDUPP, L, NNF2, NPF2
      COMMON / OUTCM / ICMW, IDBG, NPSTW, MATW, IWCF
      COMMON / LM    / MAXDIM, NMAT, MAXST, IFL, NDRAC, LVEC
      DIMENSION   ISTN(1), ISTP(1)
C****
C*****                                                   *****
C***     D E F I N I T I O N   O F   V A R I A B L E S     ***
C*****                                                   *****
C***
C***           N P B S         BASE ADDRESS (DN,DP)-PAIR
C***                             FOR FIXED NDN + NDP
C***
C***               N D N       NUM OF DN
C***               N D P       NUM OF DP
C***
      IF(IWCF.GT.0) WRITE(6,640)
      NDUPT = MIN( NDUPTX, NNF2 + NPF2 )
      NDUPN = NNF2
      NDUPP = NPF2
      GOTO  3
1     NDUPT = NDUPT - 1
3     IF( NDUPN .GT. NDUPT )  NDUPN = NDUPT
      IF( NDUPP .GT. NDUPT )  NDUPP = NDUPT
      IF( NDUPN .GT. MAXND )  NDUPN = MAXND
      IF( NDUPP .GT. MAXND )  NDUPP = MAXND
      IF(IWCF.GT.0) WRITE(6,611) NDUPT,NDUPN,NDUPP,NNF2,NPF2,L
      IT = 0
      NST = 0
      INP = 0
      NST1 = 0
      NST2 = 0
      DO 201  NDTX = 0, NDUPT
      NDNX = NDTX
    2 NDPX = NDTX - NDNX
      IF( NDPX .GT. NDUPP )  GO  TO  10
      IF( NDNX .GT. NDUPN )  GO  TO  20
      ILN=1
      IRN=NBBAS(NDNX+1)
      IF(NDNX.GT.0) ILN=NBBAS(NDNX)+1
      ILP=1
      IRP=NBBAS(NDPX+1)
      IF(NDPX.GT.0) ILP=NBBAS(NDPX)+1
      DO 203  ISTNX = ILN, IRN
      DO 202  ISTPX = ILP, IRP
      LN = LD( ISTNX )
      LP = LD( ISTPX )
      IF( IABS(LN-LP) .GT. L .OR. L .GT. LN+LP )  GO  TO  202
      NST = NST + 1
C***
      IF( NST .GT. MAXST )  GOTO  1
C***
      ISTN( NST ) = ISTNX
      ISTP( NST ) = ISTPX
202   CONTINUE
203   CONTINUE
      IF( NST .EQ. NST2 )  GOTO  20
      NST2 = NST
      INP = INP + 1
      NPBS( INP + 1 ) = NST
C***
      IF( INP .GT. 199 )  GOTO  1
C***
      NDN( INP ) = NDNX
      NDP( INP ) = NDPX
   20 NDNX = NDNX - 1
      IF( NDNX .GE. 0 )  GOTO  2
   10 IF( NST .EQ. NST1 )  GOTO  201
      NST1 = NST
      IT = IT + 1
        NDTBS( IT + 1 ) = INP
        NDT( IT ) = NDTX
  201 CONTINUE
      NSTF = NST
        INPF = INP
        ITF = IT
      DO 210  J = 1, NDUPT + 1
      DO 211  K = 1, ITF
      IF( NDT(K) .EQ. J-1 )  GO  TO  8
  211 CONTINUE
      NQBS(J) = NQBS(J-1)
      GO  TO  210
    8 KL = NDTBS(K+1)
      NQBS(J) = NPBS(KL+1)
  210 CONTINUE
      NQBS(NDUPT+1) = NSTF
C*****                                    *****
C***          O U T P U T                   ***
C*****                                    *****
      IF(IWCF.LE.0) RETURN
      IF( NPSTW .LE. 1 )  GO  TO  5
      WRITE(6,600)
      DO 204  ITX = 1, IT
        KL = NDTBS( ITX )
        KR = NDTBS( ITX + 1 )
        IF( KL . EQ. KR )  GO  TO  204
        KL = KL + 1
        DO 205  K = KL, KR
          JL = NPBS( K )
          JR = NPBS( K + 1 )
          IF( JL .EQ. JR )  GO  TO  205
          JL = JL + 1
          DO 206  J = JL, JR
            INX = ISTN( J )
            IPX = ISTP( J )
206         WRITE(6,604)  J, NDT(ITX), NDN(K), NDP(K), INX, IPX,
     1                    LD(INX), LD(IPX), NSEN(INX), NSEN(IPX)
205     CONTINUE
204   CONTINUE
5     WRITE(6,630)  NSTF
      IF( NPSTW .LE. 0 )  RETURN
      WRITE(6,612)  ( NQBS( ITX ), ITX = 1, NDUPT + 1 )
      RETURN
C*****                                    *****
C***               F O R M A T              ***
C*****                                    *****
  630 FORMAT( //T7,'NUM  OF  STATES  =',I5 )
  640 FORMAT( '1' / )
  611 FORMAT( T3, '*****   P R O B L E M   *****' //
     1  T10, 'UPPER  LIMIT  OF  TOTAL    D-BOSON  #  =',I3 /
     2  T10, '.....  .....  ..  NEUTRON  .......  .  =',I3 /
     3  T10, '.....  .....  ..  PROTON   .......  .  =',I3 /
     4  T10, 'TOTAL  NEUTRON  BOSON  #  =',I3 /
     5  T10, '.....  PROTON   .....  .  =',I3 //
     6  T10, 'TOTAL  ANGULAR  MOMENTUM  =',I3 )
  600 FORMAT( //
     1  T7,'*****   C O N F I G U R A T I O N   T A B L E   *****'///
     2  T9,'PAIR-NUM',3(3X,'N OF',3X),3(3X,'NEUTRON',3X,'PROTON ')/
     3  T20,'TOTAL',5X,'NEUTRON',3X,'PROTON',T50,2('STATE',5X),
     4  T70,2('ANGULAR',3X),T90,2('O(5)',6X)/
     5  T20,3('D-BOSON',3X),T50,2('NUMBER',4X),T70,2('MOM.',6X),
     6  T90,2('SEN.',6X)/)
  604 FORMAT( T5, 12I10 )
  612 FORMAT( ///  T7, '*****   N Q B S   *****'//( T12,10I10 ) )
      END
      SUBROUTINE  B S Y S M E  ( WAV, IHBASE, ISTN, ISTP )
C *******************************************************
C *****              V - P   VERSION                *****
C *******************************************************
      IMPLICIT  REAL*8(A-H,O-Z)
      PARAMETER  ( ICFP1=2907, ICFP2=6371, MAXHMF=256 )
C **********************************************************************
      PARAMETER   ( IHSTR=80000 )
      COMMON / HMLCM / H(IHSTR), JIND(IHSTR)
      COMMON / SPSTCM /  ISTBS(300),NBD(300),LD(300),NSEN(300),NDL(300)
     1                  ,NBBAS(16),ISTBS2(300),LAD(300),LADBS(16)
      COMMON / CFP1CM / IDAU(ICFP1),L1(ICFP1),CFP1(ICFP1)
      COMMON / CFP2CM / IDAU2(ICFP2),L2(ICFP2),CFP2(ICFP2),LIM(ICFP2)
      COMMON / NPST / NDN(200),NDP(200),NPBS(200),
     1                NDT(20),NDTBS(20),NQBS(-3:20),NST
      COMMON / SR /   SR(0:500)
      EQUIVALENCE  (SQ5,SR(5))
      COMMON / NBCNT / NDUPT, NDUPN, NDUPP, L, NNF2, NPF2
      COMMON / H / EFIX,
     1             ESPN, C0N, C2N, C4N, VYN, VWN, VAN, VBN, VCN,
     2             ESPP, C0P, C2P, C4P, VYP, VWP, VAP, VBP, VCP,
     3             VAV, VXV, VVX, VXX(0:4), VVV,
     5             SNSP, DNSP, SNDP, DNDP, VSSN, VSSP,
     &             SNSN,SNDN,DNDN,SPSP,SPDP,DPDP,ESN,ESP
      COMMON / EQQ   / FEQ(2),
     1     QCN, QSPN, QC0N, QC2N, QC4N, QVYN, QVWN, QVAN, QVBN, QVCN,
     2     QCP, QSPP, QC0P, QC2P, QC4P, QVYP, QVWP, QVAP, QVBP, QVCP
      COMMON / OUTCM / ICMW, IDBG, NPSTW, MATW, IWCF
      COMMON / LM    / MAXDIM, NMAT, MAXST, IFL, NDRAC, LVEC
      COMMON / HFL   / NMEFL(MAXHMF), IRFL(MAXHMF)
C****
      DIMENSION     WAV(1), ISTN(1), ISTP(1), IHBASE(1)
C
      AND1(I) = IAND(1,I)
      RACL( M, J, K, I ) = RACAR( M, J, K, I )
C #   RACL( M, J, K, I ) = DRAC( M*2, J*2, K*2, I*2, L*2, 4 )
      SQH = DSQRT( 0.5D0 )
 
C********************************************************************
      IF( NDRAC .GE. 1 )  CALL  RACIN  ( L )
C********************************************************************
      ICKCFP = 0
      IH   = 0
      NMAT = 0
      IFL  = 0
      FNN = NNF2
      FNP = NPF2
 
      IHBASE( 1 ) = 0
      DO 200  I = 1, NST
 
      DO 208  K = 1, I
        WAV( K ) = 0
208   CONTINUE
      IN  = ISTN( I )
      IP  = ISTP( I )
      LN  = LD  ( IN )
      LP  = LD  ( IP )
      LN2 = 2*LN
      LP2 = 2*LP
      NN  = NBD ( IN )
      NSBN = NNF2 - NN
      DBN = NN
      NP  = NBD( IP )
      NSBP = NPF2 - NP
      DBP = NP
      NT  = NN + NP
      NSN = NSEN( IN )
      SN  = NSN
      NSP = NSEN( IP )
      SP = NSP
      FLN = LN * ( LN + 1 )
      FLP = LP * ( LP + 1 )
      FSN = FNN - DBN
      FSP = FNP - DBP
C*****                                                      *****
C***          D I A G O N A L   P A R T S                     ***
C*****                                                      *****
      HM = DNSP * FSP * DBN + SNDP * DBP * FSN + DNDP * DBN * DBP
     &   + SNSP * FSN * FSP
     &  + SNSN*FSN*FSN + SNDN*FSN*DBN + DNDN*DBN*DBN + ESN*FSN
     &  + SPSP*FSP*FSP + SPDP*FSP*DBP + DPDP*DBP*DBP + ESP*FSP
      WAV( I ) = EFIX + HM
     &         + VSSN * ( ( NSBN * ( NSBN - 1 ) ) / 2 )
     &         + VSSP * ( ( NSBP * ( NSBP - 1 ) ) / 2 )
     &         + DBN * ( ESPN + ( VAN * ( DBN - 1 ) ) / 2 )
     &         + VBN * ( DBN - SN ) * ( DBN + SN + 3 )
     &         + VCN * ( FLN - 6 * DBN )
     &         + DBP * ( ESPP + ( VAP * ( DBP - 1 ) ) / 2 )
     &         + VBP * ( DBP - SP ) * ( DBP + SP + 3 )
     &         + VCP * ( FLP - 6 * DBP )
      WAV( I ) = WAV( I )
     &         + DBP * QCN + DBN * QCP
     &         + DBP * ( DBN * ( QSPN + ( QVAN * (DBN-1) ) / 2 )
     &                 + QVBN * (DBN-SN) * (DBN+SN+3)
     &                 + QVCN * (FLN-6*DBN) )
     &         + DBN * ( DBP * ( QSPP + ( QVAP * (DBP-1) ) / 2 )
     &                 + QVBP * (DBP-SP) * (DBP+SP+3)
     &                 + QVCP * (FLP-6*DBP) )
C*****                                                      *****
C***          H Y   F O R   N E U T R O N                     ***
C*****                                                      *****
      IF( VYN .EQ. 0 .AND. QVYN .EQ. 0 )  GO  TO  10
      IL = ISTBS2( IN )
      IR = ISTBS2( IN + 1 )
      IF( IL .EQ. IR )  GO  TO  10
      IL = IL + 1
      DO 202  IA = IL, IR
      IAX = IA
      IF( LIM( IA ) - 2 )  202, 12, 10
  202 CONTINUE
      GO  TO  10
   12 DO 203  IA = IAX, IR
      IA1 = IA
      IF( LIM( IA ) .NE. 2 )  GO  TO  13
  203 CONTINUE
      IA1 = IR + 1
   13 IR = IA1 - 1
      KL = NQBS( NT - 1 )
      KR = NQBS( NT )
      IF( KL .EQ. KR )  GO  TO  10
      KL1 = KL + 1
      DO 201  K = KL1, KR
      KZ = ISTN( K )
      IF( NBD( KZ ) .NE. ( NN - 1 ) )  GO  TO  201
      IF( LD( KZ ) .NE. LN )    GO  TO  201
      IF( ISTP( K ) .NE. IP )  GO  TO  201
      IA = IAX
      HYN = 0
      JA = ISTBS( KZ ) + 1
      JR = ISTBS( KZ + 1 )
   14 IB = IDAU2( IA )
   15 JB = IDAU( JA )
   16 IF( IB - JB )  17, 18, 19
   19 JA = JA + 1
      IF( JA .LE. JR )  GO  TO  15
      GO  TO  20
   17 IA = IA + 1
      IF( IA .GT. IR )  GO  TO  20
      IB = IDAU2( IA )
      GO  TO  16
   18 HYN = HYN + CFP2( IA ) * CFP1( JA )
      JA = JA + 1
      IA = IA + 1
      IF( JA .LE. JR .AND. IA .LE. IR )  GO  TO  14
   20 WAV( K ) = WAV( K )
     1       + ( VYN + DBP * QVYN ) * ( DBN - 1 ) * HYN
     2         * SQH * SR( NN ) * SR( NNF2 + 1 - NN )
  201 CONTINUE
C*****                                                      *****
C***          H Y   F O R   P R O T O N                       ***
C*****                                                      *****
   10 IF( VYP .EQ. 0 .AND. QVYP .EQ. 0 )  GO  TO  30
      IL = ISTBS2( IP )
      IR = ISTBS2( IP + 1 )
      IF( IL .EQ. IR )  GO  TO  30
      IL = IL + 1
      DO 302  IA = IL, IR
      IAX = IA
      IF( LIM( IA ) - 2 )  302, 32, 30
  302 CONTINUE
      GO  TO  30
   32 DO 303  IA = IAX, IR
      IA1 = IA
      IF( LIM( IA ) .NE. 2 )  GO  TO  33
  303 CONTINUE
      IA1 = IR + 1
   33 IR = IA1 - 1
      KL = NQBS( NT - 1 )
      KR = NQBS( NT )
      IF( KL .EQ. KR )  GO  TO 30
      KL1 = KL + 1
      DO 301  K = KL1 , KR
      KZ = ISTP( K )
      IF( NBD( KZ ) .NE. ( NP - 1 ) )  GO  TO  301
      IF( LD( KZ ) .NE. LP )    GO  TO  301
      IF( ISTN( K ) .NE. IN )  GO  TO  301
      IA = IAX
      HYP = 0
      JA = ISTBS( KZ ) + 1
      JR = ISTBS( KZ + 1 )
   34 IB = IDAU2( IA )
   35 JB = IDAU( JA )
   36 IF( IB - JB )  37, 38, 39
   39 JA = JA + 1
      IF( JA .LE. JR )  GO  TO  35
      GO  TO  40
   37 IA = IA + 1
      IF( IA .GT. IR )  GO  TO  40
      IB = IDAU2( IA )
      GO  TO  36
   38 HYP = HYP + CFP2( IA ) * CFP1( JA )
      JA = JA + 1
      IA = IA + 1
      IF( JA .LE. JR .AND. IA .LE. IR )  GO  TO  34
   40 WAV( K ) = WAV( K )
     1         + ( VYP + DBN * QVYP ) * ( DBP - 1 ) * HYP
     2           * SQH * SR( NP ) * SR( NPF2 + 1 - NP )
  301 CONTINUE
C*****                                                      *****
C***          H W   F O R   N E U T R O N                     ***
C*****                                                      *****
   30 IF( VWN .EQ. 0 .AND. QVWN .EQ. 0 )  GO  TO  50
      IF( NN .EQ. NSN )  GO  TO  50
      KL = NQBS( NT - 2 )
      KR = NQBS( NT - 1 )
      IF( KL .EQ. KR )  GO  TO  50
      IR = ISTBS2( IN + 1 )
      IL = ISTBS2( IN ) + 1
      DO 205  IA = IL, IR
      IF( LIM( IA ) .NE. 0 )  GO  TO  50
      IB = IDAU2( IA )
      IF( NSEN( IB ) .NE. NSN )  GO  TO  205
      KL1 = KL + 1
      DO 206  K = KL1, KR
      IF( IB .EQ. ISTN( K ) .AND. IP .EQ. ISTP( K ) )  GO  TO  52
  206 CONTINUE
      GO  TO  205
   52 WAV( K ) = ( VWN + DBP * QVWN ) * 0.5D0 * CFP2( IA )
     2         * SR( NN ) * SR( NN - 1 ) * SR( NNF2 + 2 - NN )
     3         * SR( NNF2 + 1 - NN )
  205 CONTINUE
C*****                                                      *****
C***          H W   F O R   P R O T O N                       ***
C*****                                                      *****
   50 IF( VWP .EQ. 0 .AND. QVWP .EQ. 0 )  GOTO  60
      IF( NP .EQ. NSP )  GO  TO  60
      IR = ISTBS2( IP + 1 )
      IL = ISTBS2( IP ) + 1
      KL = NQBS( NT - 2 )
      KR = NQBS( NT - 1 )
      IF( KL .EQ. KR )  GO  TO  60
      DO 305  IA = IL, IR
      IF( LIM( IA ) .NE. 0 )  GO  TO  60
      IB = IDAU2( IA )
      IF( NSEN( IB ) .NE. NSP )  GO  TO  305
      KL1 = KL + 1
      DO 306  K = KL1, KR
      IF( IB .EQ. ISTP( K ) .AND. IN .EQ. ISTN( K ) )  GO  TO  62
  306 CONTINUE
      GO  TO  305
   62 WAV( K ) = WAV( K )
     1         + ( VWP + DBN * QVWP ) * 0.5D0 * CFP2( IA )
     2           * SR( NP ) * SR( NP - 1 ) * SR( NPF2 + 2 - NP )
     3           * SR( NPF2 + 1 - NP )
  305 CONTINUE
   60 CONTINUE
C*****                                                             *****
C***          H V V   F O R   N E U T R O N - P R O T O N            ***
C*****                                                             *****
      IF( VVV .EQ. 0 )  GO TO 100
      IF( NN .EQ. 0 .OR. NP .EQ. 0 )  GO  TO  150
      KL = NQBS( NT - 2 )
      KR = NQBS( NT - 1 ) + 1
      IF( KL .EQ. KR )  GO TO 100
      IL = ISTBS( IN ) + 1
      IR = ISTBS( IN + 1 )
      JL = ISTBS( IP ) + 1
      JR = ISTBS( IP + 1 )
      KL1 = KL + 1
      DO 210  K = KL1, KR
      KN = ISTN( K )
      IF( NBD( KN ) .NE. NN - 1 )  GO TO 210
      IF( IABS(LD(KN)-LN) .GT. 2 .OR. LD(KN)+LN .LT. 2 )  GO  TO  210
      KP = ISTP( K )
      IF( NBD( KP ) .NE. NP - 1 )  GO TO 210
      IF( IABS(LD(KP)-LP) .GT. 2 .OR. LD(KP)+LP .LT. 2 )  GO  TO  210
*VOCL LOOP,VECTOR
      DO 211  IXX = IL, IR
        IF( IDAU( IXX ) .EQ. KN )  GOTO 1211
  211 CONTINUE
      ICKCFP = 211
1211  CFPX1 = CFP1( IXX )
*VOCL LOOP,VECTOR
      DO 212  JXX = JL, JR
        IF( IDAU( JXX ) .EQ. KP )  GOTO 1212
  212 CONTINUE
      ICKCFP = 212
1212  CFPX2 = CFP1( JXX )
      HVV = SR( NN * ( LN2 + 1 ) ) * SR( NP * ( LP2 + 1 ) )
     1    * CFPX1 * CFPX2 * VVV
     2    * RACL( LN, LP, LD(KN), LD(KP) )
     3    * SR( NNF2 + 1 - NN ) * SR( NPF2 + 1 - NP )
      IF( AND1( LD(KN) + LP + L ) .EQ. 1 ) HVV = - HVV
      WAV( K ) = HVV
  210 CONTINUE
C*****                                                             *****
C***          H X V   F O R   N E U T R O N - P R O T O N            ***
C*****                                                             *****
  100 IF( VXV .EQ. 0 )  GO TO 73
      KL = NQBS( NT - 1 )
      KR = NQBS( NT )
      IF( KL .EQ. KR )  GO TO 73
      IL = ISTBS( IN ) + 1
      IR = ISTBS( IN + 1 )
      JL = ISTBS( IP ) + 1
      JR = ISTBS( IP + 1 )
      KL1 = KL + 1
      DO 214  K = KL1, KR
      KP = ISTP( K )
      IF( NBD(KP) .NE. NP - 1 )  GO TO 214
      IF( IABS(LD(KP)-LP) .GT. 2 .OR. LD(KP)+LP .LT. 2 )  GO  TO  214
      KN = ISTN( K )
      IF( NBD( KN ) .NE. NN )  GO TO 214
      IF( IABS(LD(KN)-LN) .GT. 2 .OR. LD(KN)+LN .LT. 2 )  GO  TO  214
      DO 215  JX3 = JL, JR
        IF( IDAU( JX3 ) .EQ. KP )  GOTO  70
215   CONTINUE
      ICKCFP = 215
70    CFPX1 = CFP1( JX3 )
      HXV = DDMAT( KN, IN, 2 )
     1    * SR( NP * ( LP2 + 1 ) ) * CFPX1 * VXV
     2    * RACL( LN, LP, LD(KN), LD(KP) )
     3    * SR( NPF2 + 1 - NP )
      IF( AND1( L + LP + LD(KN) ) .EQ. 1 )  HXV = - HXV
C----------------IF(  OMGN .EQ. 0 )  GO  TO  120
C----------------HXV = HXV * ( OMGN - 2 ) / ( OMGN - 2 * DBN )
      WAV( K ) = WAV( K ) + HXV
  214 CONTINUE
C*****                                                             *****
C***          H V X   F O R   N E U T R O N - P R O T O N            ***
C*****                                                             *****
   73 IF( VVX .EQ. 0 )  GO TO 93
      KL = NQBS( NT - 1 )
      KR = NQBS( NT )
      IF( KL .EQ. KR )  GO TO 93
      JL = ISTBS( IN ) + 1
      JR = ISTBS( IN + 1 )
      IF( JL .GT. JR )  GO  TO  93
      IL = ISTBS( IP ) + 1
      IR = ISTBS( IP + 1 )
      IF( IL .GT. IR )  GO  TO  93
      KL1 = KL + 1
      DO 314  K = KL1, KR
      KN = ISTN( K )
      IF( NBD(KN) .NE. NN - 1 )  GO TO 314
      IF( IABS(LD(KN)-LN) .GT. 2 .OR. LD(KN)+LN .LT. 2 )  GO  TO  314
      KP = ISTP( K )
      IF( NBD( KP ) .NE. NP )  GO TO 314
      IF( IABS(LD(KP)-LP) .GT. 2 .OR. LD(KP)+LP .LT. 2 )  GO  TO  314
      DO 315  JX = JL, JR
        IF( IDAU( JX ) .EQ. KN )  GOTO  90
  315 CONTINUE
      ICKCFP = 315
   90 HVX = DDMAT( KP, IP, 2 )
     1    * SR( NN * ( LN2 + 1 ) ) * CFP1( JX ) * VVX
     2    * RACL( LN, LP, LD(KN), LD(KP) )
     3    * SR( NNF2 + 1 - NN )
      IF( AND1( L + LP + LD(KN) ) .EQ. 1 )  HVX = - HVX
C-----------HVX = HVX * ( OMGP - 2 ) / ( OMGP - 2 * DBP )
      WAV( K ) = WAV( K ) + HVX
  314 CONTINUE
C*****                                                             *****
C***          H X X   F O R   N E U T R O N - P R O T O N            ***
C*****                                                             *****
   93 KL = NQBS( NT )
      IL = ISTBS( IN ) + 1
      IR = ISTBS( IN + 1 )
      JL = ISTBS( IP ) + 1
      JR = ISTBS( IP + 1 )
      DO 230  M = 0, 4
        IF( VXX(M) .EQ. 0 ) GOTO 230
        KL1 = KL + 1
        DO 220  K = KL1, I
          KP = ISTP( K )
          IF( NBD(KP) .NE. NP )  GO TO 220
          IF( IABS(LD(KP)-LP) .GT. M .OR. LD(KP)+LP .LT. M )  GOTO  220
          KN = ISTN( K )
          IF( NBD(KN) .NE. NN )  GO TO 220
          IF( IABS(LD(KN)-LN) .GT. M .OR. LD(KN)+LN .LT. M )  GOTO  220
          HXX = VXX( M ) * DDMAT( KN, IN, M ) * DDMAT( KP, IP, M )
     &        * DRACLM( LN, LP, LD(KN), LD(KP), L, M )
          IF( AND1( ( L + LP + LD(KN) ) ) .EQ. 1 )  HXX = - HXX
C**   IF( OMGN .EQ. 0 )  GO  TO  140
C**   HXX = HXX * ( OMGN - 2 ) / ( OMGN - DBN )
C*140 IF( OMGP .EQ. 0 )  GO  TO  142
C**   HXX = HXX * ( OMGP - 2 ) / ( OMGP - DBP )
          WAV( K ) = WAV( K ) + HXX
220     CONTINUE
230   CONTINUE
C*****                                                             *****
C***          H A V   F O R   N E U T R O N - P R O T O N            ***
C*****                                                             *****
  150 IF( VAV .EQ. 0 )  GO  TO  280
      IF( NP .LE. 0 )  GO  TO  280
      IL = ISTBS( IP ) + 1
      IR = ISTBS( IP + 1 )
      KL = NQBS( NT )
      IF( KL .EQ. I  )  GO  TO  280
      KL1 = KL + 1
      DO 260  K = KL1, I
      KN = ISTN( K )
      IF( NBD( KN ) .NE. NN + 1 )  GO TO 260
      IF( IABS(LD(KN)-LN) .GT. 2 .OR. LD(KN)+LN .LT. 2 )  GO  TO  260
      KP = ISTP( K )
      IF( NBD( KP ) .NE. NP - 1 )  GO TO 260
      IF( IABS(LD(KP)-LP) .GT. 2 .OR. LD(KP)+LP .LT. 2 )  GO  TO  260
      DO 261  IX = IL, IR
        IF( IDAU( IX ) .EQ. KP )  GOTO  161
  261 CONTINUE
      ICKCFP = 261
  161 CFPX1 = CFP1( IX )
      JL = ISTBS( KN ) + 1
      JR = ISTBS( KN + 1 )
      DO 262  JX = JL, JR
        IF( IDAU( JX ) .EQ. IN )  GOTO  163
  262 CONTINUE
      ICKCFP = 262
  163 CFPX2 = CFP1( JX )
      LDN = LD(KN)
      LDN2 = LDN*2
      LDP = LD(KP)
      HAV = SR( NP*(NN+1) ) * SR( LP2 + 1 ) * SR( LDN2 + 1 )
     1    * CFPX1 * CFPX2 * VAV
     2    * RACL( LN, LP, LDN, LDP )
     3    * SR( NPF2 + 1 - NP ) * SR( NNF2 - NN )
      IF( AND1( LN  +  LP   + L ) .EQ. 1 )  HAV = - HAV
      WAV( K ) = WAV( K ) + HAV
  260 CONTINUE
C***
      IF( ICKCFP .NE. 0 )  THEN
        WRITE(6,'(T5,''STOP IN BSYSME #ICKCFP ='',I6)' )
        STOP 'BSYSME ICKCFP ERROR'
      ENDIF
280   WAV( I ) = WAV( I ) / 2
400     DO 194  K = 1, I
          IF( WAV( K ) .EQ. 0 )  GOTO  194
          IH = IH + 1
          IF( IH .GT. MAXDIM ) GOTO 410
          JIND( IH ) = K
          H   ( IH ) = WAV( K )
194     CONTINUE
        IHBASE( I + 1 ) = IH
        NME = IH
        GOTO  200
410     IF( IFL .EQ. 0 )  THEN
          REWIND  3
          IRFL( 1 ) = 1
        ENDIF
        IF( IFL .GE. MAXHMF )  GOTO  8
        WRITE( 3 )   JIND,  H
CC      WRITE( 3 ) ( JIND( IF ), IF = 1, NME ),
CC   &             ( H   ( IF ), IF = 1, NME )
        IFL = IFL + 1
        NMEFL( IFL ) = NME
        IRFL ( IFL + 1 ) = I
        NMAT = NMAT + NME
        IH = 0
        NME = 0
        GOTO  400
 
200   CONTINUE
 
      NMAT = NMAT + IH
      IF( IH .EQ. 0 )  GOTO  6
      IF( IFL .EQ. 0 )  GOTO  7
        IF( IFL .GE. MAXHMF )  GOTO  8
      WRITE( 3 )    JIND,  H
CC    WRITE( 3 )  ( JIND( IF ), IF = 1, IH ),
CC   &            ( H   ( IF ), IF = 1, IH )
      IFL = IFL + 1
      NMEFL( IFL ) = IH
6     IF( IFL .NE. 0 )  R E W I N D   3
7     HXX = DFLOAT(NMAT*100)/DFLOAT(((NST+1)*NST)/2)
      K = ((NMAT-1) / 1589) + 1
      IF( IFL .EQ. 0 )  K = 0
      WRITE(6,613)  NMAT, HXX, K, IFL
C      IF( MATW .EQ. 0 )  R E T U R N
C      IF( IFL .GT. 0 )  R E T U R N
C      WRITE(6,620)  ( IFH(1,I), IFH(2,I), H(I), I = 1, NMAT )
      R E T U R N
8     STOP 'IN BSYSME : INCREASE MAXHMF'
613   FORMAT(
     &  /T7,'# OF M.E. =',I8,5X,'NON-ZERO / TRI-ANGLE =',F5.1,' (%)'
     &  /T7,'FILE ACCESS (#3) =',I6,' (/19.1KB)',5X,'LOGICAL =',I4 )
  620 FORMAT((T4,6(2I5,F8.3,2X)))
      END
      SUBROUTINE  E I G L A N  ( WAV, VEC, ICODE, IHBASE,
     &                           ISTN, ISTP )
C******************************************
C *****      V  - P    VERSION      ******
C******************************************
      IMPLICIT  REAL*8(A-H,O-Z)
 
C16      REAL*16   A,B,B2,FS,E,G,H,X,X2,Y,QT,QS,Z,FDIM
C     INTEGER*2  IFH
      CHARACTER    IVER*4, ITITLE*4, DAYV*8, TMV*8
      PARAMETER   ( NEST=50, NTR=100, MAXHMF=256, KLMAX=600 )
 
      PARAMETER   ( IHSTR=80000 )
      COMMON / HMLCM / HINT(IHSTR), JIND(IHSTR)
 
      DIMENSION   WAV(1), VEC(1), IHBASE(1),
     &            ISTN(1), ISTP(1)
      COMMON / HFL  / NMEFL(MAXHMF), IRFL(MAXHMF)
      COMMON / SPSTCM /  ISTBS(300),NBD(300),LD(300),NSEN(300),NDL(300)
     1                  ,NBBAS(16),ISTBS2(300),LAD(300),LADBS(16)
      COMMON / NPST / NDN(200),NDP(200),NPBS(200),
     1                NDT(20),NDTBS(20),NQBS(-3:20),NST
      COMMON / NBCNT / NDUPT, NDUPN, NDUPP, LL, NNF2, NPF2
      COMMON / OUTCM / ICMW, IDBG, NPSTW, MATW, IWCF
      COMMON / LM    / MAXDIM, NMAT, MAXST, IFL, NDRAC, LVEC
      PARAMETER   ( NEMAX=50 )
      COMMON / QQEXCM / CHN, CHP, EXQ (NEMAX,NEMAX,3),
     &                  EXN (NEMAX,NEMAX,2), EXF (NEMAX,NEMAX)
      COMMON / LAN  / NENRGY, NVEC, R, LFILE, NCUT1, EFIX, ISYM,
     &                IFILE, TRVASM
      COMMON / EIG  / A(NTR), B(NTR), FS(NTR), EIG(NEST), N, MEIG
      COMMON / Evec / nlcz(NEST)
      COMMON / O    / ENERGY(KLMAX), JSTAT(KLMAX), FSTST(KLMAX),
     &                KLEVEL, KLVL1
      COMMON / VERF / IVER(10), ITITLE(2), DAYV(2), TMV
 
      DIMENSION  B2(NTR), EIG1(NEST), ND(2)
 
      DATA    EN, EPS / 0.D0, 0.D0 /
      DATA    SAFE, SS, S1 / 1.D-3, 1.D-12, 1.D-5 /
 
C***
C**    SAFE  :  MINIMAL  ALLOWED  LEVEL  SPACING
C**    SS    :  SMALLEST  ALLOWED  COMPONENT  IN  EIGENVECTOR
C***
      KFILE = 13
      MEIG  = NENRGY
      N     = 0
      IFLG  = 0
      IDFLG = 0
      LVEC  = IABS( NVEC )
      IF( LVEC .GT. MEIG )  LVEC = MEIG
C
      IH = 1
      HN = 10
      EN1 = 10
      NCUT = NCUT1
      IF ( NCUT .GT. NST )  NCUT = NST
C     ******
      NXC = NDUPT + 1
      IF( NVEC .EQ. 0 )  GO  TO  3
      WRITE( LFILE )  NDUPT, NDUPN, NDUPP, LL, NNF2, NPF2
      WRITE( LFILE )  ICODE, IVER, ITITLE, DAYV, TMV, CHN, CHP
      WRITE( LFILE )  NXC, ( NQBS( I ), I = 1, NXC )
      WRITE( LFILE )  NST, ( ISTN( I ), I = 1, NST ),
     1                     ( ISTP( I ), I = 1, NST )
 
3     REWIND  KFILE
 
      IF( ISYM .EQ. 0 )  GOTO  130
 
      DO 170  IC = 1, NST
170   VEC( IC ) = 0
      do 172  ic = 1, meig
172   nlcz( ic ) = ncut
 
      KSYM = 0
  270 ISYMAB = IABS( ISYM ) + KSYM
      DO 110  I = 1, NDUPT + 1
      IF( NQBS( I ) .EQ. 0 )  GO  TO  110
      ISYMAB = ISYMAB - 1
      IF( ISYMAB .EQ. 0 )  GO  TO  120
  110 CONTINUE
      STOP  'EIGLAN  ST# 110'
  120 IA = NQBS( I - 1 ) + 1
      IB = NQBS( I )
      FDIM = 2
      FDIM = 1 / SQRT( FDIM )
      IF( IA .LE. IB ) GOTO 260
  290 CONTINUE
      KSYM = KSYM + 1
      GO  TO  270
  260 DO 250  KA = IA, IB
      IF( ISTN(KA) .EQ. ISTP(IA) .AND. ISTP(KA) .EQ. ISTN(IA) )GO TO 280
  250 CONTINUE
      GO  TO  40
  280 IB = KA
      VECIB = FDIM
      IF( ISYM .LT. 0 )  VECIB = - VECIB
      LIA = LD( ISTN(IA) )
      LIB = LD( ISTP(IA) )
      IF( IAND(LIA+LIB+LL,1) .EQ. 1 ) VECIB = - VECIB
      IF( ( IA.EQ.IB ) .AND. ( FDIM .EQ. - VECIB ) )  GOTO  290
      IF( IA .EQ. IB )  GOTO  40
      VEC( IA ) = FDIM
      VEC( IB ) = VECIB
      IF( IWCF .LE. 0 )  GOTO  44
      WRITE(6,610)  IA, VEC(IA), IB, VEC(IB)
      GOTO  44
40    IF( IA .LT. NST )  THEN
        KSYM = KSYM + 1
        GOTO  270
      ELSE
        VEC(IA) = 1
        IF(IWCF.GT.0) WRITE(6,611) IA
      ENDIF
 
      GOTO  44
 
130   FN = 0
      DO 230  I = 0, NDUPT
        F = ( NDUPT + 1 - I )
        FSQ = F ** 2
        FSL = F * TRVASM
        FSLSQ = FSL ** 2
        IA = NQBS( I ) + 1
        IB = NQBS( I + 1 )
        IF( IA .GT. IB )  GOTO  230
        DO 232  J = IA, IB
          IF( NBD( ISTN(J) ) .GE. NBD( ISTP(J) ) )  THEN
            VEC( J ) = F
            f = - f
            FN = FN + FSQ
          ELSE
            VEC( J ) = FSL
            FN = FN + FSLSQ
          ENDIF
232     CONTINUE
230   CONTINUE
      F = 1 / SQRT( FN )
      DO 234  J = 1, NST
        VEC( J ) = VEC( J ) * F
234   CONTINUE
 
C**********************************************************************
 
C     THE 1-ST VECTOR OF LANCZOS BASE  WRITE-OUT ON FILE - KFILE
 
44    WRITE ( KFILE )  ( VEC(I),I=1, NST )
 
      IF ( N .NE. 0 )  GOTO  15
      DO  2  I = 1, NST
2     WAV(I) = 0
      GOTO  15
 
C*************  MATRIX OPERATION   ****************
 
15    IF( IFL .EQ. 0 )  THEN
      DO 400  I = 1, NST
        I1 = IHBASE( I ) + 1
        I2 = IHBASE( I + 1 )
        IF( I1 .GT. I2 )  GOTO  400
        VECI = VEC( I )
*VOCL LOOP,NOVREC(WAV)
        DO 410  K = I1, I2
          WAV( JIND(K) ) = WAV( JIND(K) ) + HINT( K ) * VECI
410     CONTINUE
        WAVI = 0
        DO 420  K = I1, I2
          WAVI = WAVI + HINT( K ) * VEC( JIND(K) )
420     CONTINUE
        WAV( I ) = WAVI + WAV( I )
400   CONTINUE
      ELSE
        REWIND  3
        KFL = 1
        DO 440  I = 1, NST
          I1 = IHBASE( I ) + 1
          I2 = IHBASE( I + 1 )
          IF( I .EQ. IRFL(KFL) )  THEN
            NME = NMEFL( KFL )
            KFL = KFL + 1
            IF( KFL .GT. IFL )  KFL = IFL
            I1 = 1
            READ( 3 )    JIND,  HINT
CC          READ( 3 )  ( JIND( IF ), IF = 1, NME ),
CC   &                 ( HINT( IF ), IF = 1, NME )
          ENDIF
          IF( I1 .GT. I2 )  GOTO  440
          VECI = VEC( I )
*VOCL LOOP,NOVREC(WAV)
          DO 450  K = I1, I2
            WAV( JIND(K) ) = WAV( JIND(K) ) + HINT( K ) * VECI
450       CONTINUE
          WAVI = 0
          DO 460  K = I1, I2
            WAVI = WAVI + HINT( K ) * VEC( JIND(K) )
460       CONTINUE
          WAV( I ) = WAVI + WAV( I )
440     CONTINUE
      ENDIF
      R E W I N D    3
C*****************************************************************
C
C      TRI-DIAGONAL  MATRIX  IS  CALCULATED  FROM  < HINT >
C         A ( I )  :  DIAGONAL  MATRIX  ELEMENT
C         B ( I )  :  BI-DIAGONAL  MATRIX  ELEMENT
C
C*****************************************************************
1     N = N + 1
      nlcz( ih ) = n
      X = 0
      DO  84  I = 1, NST
        X = X + WAV( I ) * VEC( I )
84    CONTINUE
      Y = 0
      DO  85  I = 1, NST
        Y = Y + WAV( I ) ** 2
85    CONTINUE
      A(N) = X
      Y = Y - X * X
      IF( ABS( Y ) .LE. S1 )  GOTO  100
      B2(N) = Y
      Y     = SQRT( Y )
      B (N) = Y
      DO 86  I = 1, NST
        QT = VEC( I )
        QS = WAV( I )
        WAV( I ) = - QT * Y
        IF( ABS( WAV( I ) ) .LT. SS )  WAV( I ) = 0
        VEC( I ) = QS - QT * X
        IF( ABS( VEC( I ) ) .LT. SS )  VEC( I ) = 0
86    CONTINUE
      Z = 0
      DO  88  I = 1 , NST
        Z = Z + VEC( I ) ** 2
88    CONTINUE
      IF( ABS( 1-Z ) .LT. 1.D-10 )  GOTO 93
      Z = 1 / SQRT( Z )
      DO  90  I = 1 , NST
        VEC( I ) = VEC( I ) * Z
90    CONTINUE
C
   93 CONTINUE
      GOTO   94
  100 IDFLG = 1
      GOTO  101
   94 IF(IWCF .GE. 2 )  WRITE(6,302) N,A(N),B(N),B2(N)
      IF( N .GE. NCUT )  IDFLG = 1
      IF( N .GE. NCUT )  GOTO  105
      IF( N .LT. MEIG )  GOTO  95
      GOTO   105
C
  101 IF(IWCF  .GE. 1 ) WRITE(6,303) N,X,Y
      IF( MEIG .GT. N )  MEIG = N
      do 174  i = ih, meig
174   nlcz( i ) = n
C
C
C*****      S T A R T    O F    B I S E C T I O N       ******
C
C*****
C***    WE ARE SEARCHING THE IH-TH EIGENVALUE .
C*****
  105 K = IH
      do 175  i = ih, meig
175   nlcz( i ) = n
      IF( N  .EQ. 1 )   GOTO  70
      IF( IH .GE. 2 )   GOTO  24
C*****
C***    THE LOWER BOUND IS DETERMINED BY THE GERSCHGORIN METHOD .
C***
C***    N-DIMENSION REAL SYMMETRIC MATRIX NECESSARILY HAS N REAL
C***    EIGENVALUES .
C*****
      DO 21  I = 1, N
        D = A(I)
        IF( I .NE. N )  D = D - B(I)
        IF( I .NE. 1 )  D = D - B(I-1)
        IF ( D .LT. HN )   HN = D
   21 CONTINUE
      EPS = ABS ( HN )  * R / 2
      EPS1 = EPS / 5
      G = HN - 1
      H = HN + 10
      GOTO  25
   24 G = EIG(IH-1) + SAFE
      H = EIG(IH)
   25 X = G
      L = 1
   35 ND(L) = 0
      E = A(1) - X
      IF ( E.LT.0 )  ND(L) = 1
      FS(1) = E
      FS(2) = ( (A(2)-X)*E - B2(1) ) / ABS(E)
      IF ( N .LT. 3 )   GO  TO    39
      DO  37  J = 3 , N
        Y = 1
        IF(FS(J-2).LT.0) Y=-1
        FS(J) = ( (A(J)-X)*FS(J-1) - B2(J-1)*Y ) / ABS( FS(J-1) )
        Y = Y * FS(J-1)
        IF ( Y .LT. 0 )    ND(L) = ND(L) + 1
37    CONTINUE
39    Y = FS(N) * FS(N-1)
      IF ( Y .LT. 0 )    ND(L) = ND(L) + 1
      IF (  L  .EQ.  1  )    GO  TO   46
      X2 = ( H - G )
      X = X2 / 2
      I = ND(2) - ND(1)
      IF (  I  .GT.  0  )    GO  TO   50
      G  = H
      H = H + X2
      ND(1) = ND(2)
   46 X  = H
      L = 2
      GOTO   35
   50 H  = H - X
      IF ( ( X.LT.EPS ) .AND. ( I.EQ. 1 ) )           GO  TO   55
      IF   ( X.GE.EPS1 )     GO  TO   46
   55 EIG(K) = H
      K = K + 1
      IF ( K .GT. MEIG )       GO  TO   60
      G  = H + SAFE
      H = H + 2
      GO  TO   25
60    IF( IWCF .GE. 2 )  WRITE(6,650)
      IF( IWCF .GE. 1 )  WRITE(6,510) (EIG(KI),KI=1,MEIG)
      IF( IWCF .GE. 2 )  WRITE(6,650)
      IFLG=IFLG+1
      EN = DABS ( EIG(IH) - EN1 )
           EN1 = EIG(IH)
   95 CONTINUE
C*****
C*****
      IF( IWCF .GE. 3 )  WRITE(6,620)  ( VEC( I ), I = 1, NST )
      GO  TO  30
C
C     THE ( N + 1 ) -TH LANCZOS BASE  WRITEN-OUT ON FILE-KFILE
C
   10 CONTINUE
      WRITE ( KFILE )  ( VEC(I),I=1, NST )
 
      GOTO  15
 
   30 IF( IDFLG .NE. 0 )  GOTO  350
      IF( IFLG .NE. 0 .AND. EN .LT. EPS )  GO TO 80
      GO  TO  10
   70 EIG(IH) = A(1)
      nlcz( ih ) = 1
        GOTO  60
C
80    IF(IWCF.GT.0) WRITE(6,501) IH,EIG(IH)
      nlcz( ih ) = n
      IQQ = IH
      IH  = IH + 1
C
C*****          E N D    O F    B I S E C T I O N       ******
C
      IF ( IDFLG .NE.  0   )      GO  TO   350
      IF ( IQQ   .LT. MEIG )      GO  TO     10
350   BE = - ENERGY(1)
      IF ( KLEVEL .EQ. 0 )     BE = - EIG(1)
      DO  355  I = 1 , MEIG
355   EIG1(I) = EIG(I) + BE
      WRITE(6,630)  LL, NST, BE, NDUPT, NDUPN, NDUPP
      WRITE(6,634)
      WRITE(6,640)  ( EIG (I), I = 1, MEIG )
      WRITE(6,634)
      WRITE(6,642)  ( EIG1(I), I = 1, MEIG )
      WRITE(6,632)
 
      IF ( NVEC .EQ. 0 )  GOTO  365
      IF(LVEC.GT.MEIG)  LVEC = MEIG
C******************************************************************
C****                                                           ***
      C A L L     E I G V E C   ( LVEC, WAV, VEC, ISTN, ISTP )
C****                                                           ***
C******************************************************************
365   KLVL1 = KLEVEL
      DO 370  I = 1, MEIG
        KLEVEL = KLEVEL + 1
        ENERGY(KLEVEL) = EIG(I)
        JSTAT (KLEVEL) = LL
370   CONTINUE
C--------------------------------------------------------
C***      OPERATION  ON  FILE - #KFILE  COMPLETED
C--------------------------------------------------------
      R E W I N D   K F I L E
C
      R E T U R N
C
302   FORMAT( T3,'   N =',I3, T14,'A =',F9.3,'  B =',F7.3,'  B2=',E9.2)
303   FORMAT( T3,'****  TRUNCATED  AT  N =',I3, 5X,'A =',F9.3, 5X,
     &           'B2=',E9.2 )
C
501   FORMAT( / T10,'******  CONVERGED  AT  EIG(',I3,') =',F11.4,
     &          '  ******' / )
510   FORMAT( T12, 'EIG =', ( T18,6F10.5 ))
610   FORMAT( /T3,'===  TRIAL  VECTOR  -->  VEC(',
     1  I3,' ) =', F8.5,5X,'VEC(',I3,' ) =',F8.5,
     2  10X,'THE  OTHERS = 0'  // )
611   FORMAT( /T3,'+++  TRIAL  VECTOR --->  VEC(',I3,
     &          ' ) = 1      THE  OTHERS  =  0' // )
620   FORMAT( // ( 11X, 10F12.7 ) )
630   FORMAT( T2,71('*') /T2,'***', T70,'***'
     2  /T2,'***',3X,'J =',I3,5X,'DIM =',I5,5X,'BE =',F7.3,T70,'***'
     &  /T2,'***', T70,'***'
     &  /T2,'***   TRUNCATION  :  TOTAL =',I3,5X,'NEUTRON =',
     4   I3,5X,'PROTON =',I3, T70,'***' )
632   FORMAT( T2,'***', T70,'***' /T2,71('*') )
634   FORMAT( T2,'***', T70,'***' )
640   FORMAT( T8, 'EIG      :', ( T18, 6F9.3 ) )
642   FORMAT( T8, 'EIG + BE :', ( T18, 6F9.3 ) )
650   FORMAT( T3, ' ' )
      E N D











      SUBROUTINE  E I G V E C   ( LVEC, WAV, VEC, ISTN, ISTP )
      IMPLICIT  REAL*8(A-H,O-Z)
C
C16      REAL*16   A,B,FS,E,X,SS,SUM,QT
      REAL*4       OV
      PARAMETER   ( NEST=50, NTR=100 )
      CHARACTER*4    CHST4 / '****' /, CHST4B / '----' /
      CHARACTER*7    CHST7 / '-------' /
 
      COMMON / SPSTCM /  ISTBS(300),NBD(300),LD(300),NSEN(300),NDL(300)
     1                  ,NBBAS(16),ISTBS2(300),LAD(300),LADBS(16)
      COMMON / NPST / NDN(200),NDP(200),NPBS(200),
     &                NDT(20),NDTBS(20),NQBS(-3:20),NST
      COMMON / NBCNT / NDUPT, NDUPN, NDUPP, L, NNF2, NPF2
      COMMON / OUTCM / ICMW, IDBG, NPSTW, MATW, IWCF
      COMMON / LAN   / NEIG, NVEC, EPS, LFILE, NCUT, BEFIX, ISYM,
     &                 IFILE, TRVASM
      COMMON / EIG / A(NTR), B(NTR), FS(NTR), EIG(NEST), N, MEIG
      COMMON / Evec / nlcz(NEST)
 
      DIMENSION   WAV(1), VEC(1), ISTN(1), ISTP(1)
      DIMENSION   IPROB(0:16,NEST), OV(NEST), NDTC(10)
 
      DATA   SS / 1.D-15 /
 
      KFILE = 13
 
      FS( 1 ) = 1
      IQ = 1
      REWIND   IFILE
 
C***********************************************************************
C***    BEGINNING  OF  LOOP  ABOUT  "IQ"  (SEQ #  OF  EIGENSTATES)   ***
C***********************************************************************
 
115   n = nlcz( iq )
      IF( N .EQ. 1 )  GOTO  170
      E = EIG( IQ )
      KI = N - 1
      KF = KI - 1
      FS(  N ) = SS
      FS( KI ) = ( ( E - A(N) ) / B(KI) ) * SS
      IF( N .GE. 3 )  THEN
        DO 120  KJ = 1 , KF
          KX = KI
          KI = KI-1
          FS(KI) = ( (E-A(KX)) * FS(KX) - B(KX) * FS(KX+1) ) / B(KI)
120     CONTINUE
      ENDIF
 
      SUM = 0
      DO  125  KJ = 1 , N
125   SUM = SUM + FS(KJ) ** 2
      SUM = 1 / SQRT( SUM )
      DO  130  KJ = 1 , N
130   FS(KJ) = FS(KJ) * SUM
 
170   DO  135  KI = 1 , NST
        VEC(KI) = 0
135   CONTINUE
 
      REWIND  KFILE
 
      DO  145  KI = 1 , N
 
        READ   ( KFILE )  (  WAV(KJ) , KJ = 1 , NST )
 
        DO  140  KL = 1 , NST
          VEC(KL) =  VEC(KL) + WAV(KL) * FS(KI)
140     CONTINUE
145   CONTINUE
 
      SUM = 0
      DO  150  KI = 1, NST
        SUM = SUM + VEC( KI ) ** 2
150   CONTINUE
      SUM = 1 / SQRT( SUM )
      DO  152  KI = 1, NST
        VEC( KI ) = VEC( KI ) * SUM
152   CONTINUE
 
C*********************************************************
C**                                                     **
      WRITE( LFILE )   IQ, EIG( IQ )
      WRITE( LFILE )   ( VEC(KI) , KI = 1 , NST )
 
      WRITE( IFILE )   ( VEC(KI) , KI = 1 , NST )
C**                                                     **
C*********************************************************
 
      IF( IWCF .GE. 2 .AND. IQ .EQ. 1 )  WRITE(6,640)
      IF( IWCF .GE. 2 )  WRITE(6,642)  IQ, ( FS( KI ), KI = 1, N )
 
      IF( IQ .EQ. LVEC )  GOTO  12
      IQ = IQ + 1
      GOTO   115
 
12    IQ = 0
      WRITE( LFILE )  IQ, IQ, IQ
      REWIND  KFILE
      REWIND  IFILE
 
C*********************************************************
 
      WRITE(6,610)
c      NDTMAX = 10
      NDTMAX = 12
      IF( NDTMAX .GT. NDUPT )  NDTMAX = NDUPT
      NDTMX1 = NDTMAX
c      IF( NDTMAX .LT. 8 )  NDTMX1 = 8
      IF( NDTMAX .LT. 10 )  NDTMX1 = 10
      WRITE(6,652) ( CHST4, KI = 0, NDTMX1 )
      WRITE(6,658)
      WRITE(6,652) ( CHST4, KI = 0, NDTMX1 )
      WRITE(6,650) ( KI,    KI = 0, NDTMAX )
      WRITE(6,654)
      WRITE(6,653) ( CHST4B, KI = 0, NDTMX1 )
      DO 230  KI = 1, LVEC
        READ( IFILE )  ( VEC( KJ ), KJ = 1, NST )
 
        DO 232  KP = 0, NDUPT
          KL = NQBS( KP ) + 1
          KR = NQBS( KP + 1 )
          SUM = 0
          IF( KL .GT. KR )  GOTO  20
          DO 234  KJ = KL, KR
            SUM = SUM + VEC( KJ ) ** 2
234       CONTINUE
20        IPROB( KP, KI ) = (SUM * 100 + 0.49)
232     CONTINUE
        WRITE(6,656)  KI, nlcz(ki), ( IPROB(KP, KI), KP = 0, NDTMAX )
230   CONTINUE
      WRITE(6,652) ( CHST4, KI = 0, NDTMX1 )
      REWIND  IFILE
 
C***********************************************************************
 
      IF( IWCF .GE. 1 )  THEN
        WRITE(6,610)
        WRITE(6,612)
        IWST = 10
        IF( NST .LT. IWST )  IWST = NST
        KWST = IWST
        IF( IWCF .GE. 2 )  KWST = NST
        WRITE(6,630)  IWST, KWST, NST, nlcz( ki )
        DO 220  KI = 1, IWST
          KN = ISTN( KI )
          KP = ISTP( KI )
          NDTC( KI ) = NBD( KN ) + NBD( KP )
220     CONTINUE
        WRITE(6,632)  ( NDTC( KI ), KI = 1, IWST )
        WRITE(6,633)  ( NBD( ISTN( KI ) ), KI = 1, IWST )
        WRITE(6,634)  ( LD ( ISTN( KI ) ), KI = 1, IWST )
        WRITE(6,635)  ( NBD( ISTP( KI ) ), KI = 1, IWST )
        WRITE(6,636)  ( LD ( ISTP( KI ) ), KI = 1, IWST )
        WRITE(6,614)  ( CHST7, KI = 1, IWST )
        DO 222  KJ = 1, LVEC
          READ( IFILE )  ( VEC( KI ), KI = 1, KWST )
          IF( KJ .NE. 1 )  WRITE(6,610)
!          WRITE(6,638)  KJ, ( VEC( KI ), KI = 1, KWST )
!
! The following 4 lines are added to indicate w.f. in a single column
          WRITE(6,*) KJ
          DO 229 KI = 1, KWST
             WRITE(6,*)  VEC( KI )
229       CONTINUE         
!
222     CONTINUE
        REWIND   IFILE
        WRITE(6,612)
        WRITE(6,610)
      ENDIF
 
C***********************************************************************
C****                                             *****
C**      O R T H O G O N A L I T Y   C H E C K      ***
C****                                             *****
      IF( IWCF .LE. 0 )  GOTO  30
      IF( LVEC .LE. 1 )  GOTO  30
      WRITE(6,670)  ( KI, KI = 1, LVEC )
      DO  300  KI = 1, LVEC
        DO 302  KJ = 1, KI-1
          READ( IFILE )
302     CONTINUE
        READ( IFILE )   ( VEC(KP), KP = 1 , NST )
        REWIND  IFILE
 
        DO 304  KJ = 1, LVEC
          READ( IFILE )   ( WAV(KP), KP = 1 , NST )
          SUM = 0
          DO 306  KP = 1, NST
            SUM = SUM + VEC( KP ) * WAV( KP )
306       CONTINUE
          OV( KJ ) = SUM
304     CONTINUE
        WRITE(6,672)  KI, ( OV( KJ ), KJ = 1, LVEC )
        REWIND   IFILE
300   CONTINUE
30    R E T U R N
C***********************************************************************
610   FORMAT( T3, ' ' )
612   FORMAT( T5, 19('****') )
614   FORMAT( T5, '------', 10A7 )
620   FORMAT( T7,'******   CONFIGURATION  TABLE   ******'/
     1  T7,'**',T43,'**'/ T7,'**',
     1  T13,'# OF',T24,'# OF',T34,'PROB.',T43,'**',/T7,'**',
     2  T13, 'D-BOSON',T24, 'STATE', T36, '(%)',T43,'**' /
     3  T7,'**',T43,'**' )
630   FORMAT(  T15,'WAVE  FUNCTIONS  :  ',
     &         T35,'FIRST ',I4,'  COMPONENTS  ARE  LABELLED'
     &        /T35,'FIRST ',I4,'  AMPLITUDES  ARE  PRINTED'
     &        /T35,'TOTAL  DIMENSION  =',I6 )
632   FORMAT( /T5,'ND(TOT)', T13,I3, 9(4X, I3) )
633   FORMAT(  T5,'ND (N)',  T13,I3, 9(4X, I3) )
634   FORMAT(  T5,'L  (N)',  T13,I3, 9(4X, I3) )
635   FORMAT(  T5,'ND (P)',  T13,I3, 9(4X, I3) )
636   FORMAT(  T5,'L  (P)',  T13,I3, 9(4X, I3) )
638   FORMAT(  T5,'NO.',I2, ( T11,10F7.3 ) )
640   FORMAT( //T7,'********  AMPLITUDE  IN  THE  LANCZOS  BASIS',
     &             '   ********' )
642   FORMAT( /T7,'--- NO.',I2,' ---' / ( T6, 8F9.5 ) )
650   FORMAT(  T17, 'ND', T22, 12( I2, 2X ) )
652   FORMAT(  T5, '*****************', 12A4 )
653   FORMAT(  T5, '-----------------', 12A4 )
654   FORMAT(  T5, 'ST. #', t11, 'N(lan)' )
656   FORMAT( T8, I2, T12, I2, T21, 12( I3, 1X ) )
658   FORMAT( T6,'PROBABILITY (%)  IN  VARIOUS  D-CONFIGURATIONS' )
670   FORMAT( /T5,'***  OVERLAPS  BETWEEN  EIGENSTATES  ***'
     &    / ( T12, 8( I2, 7X ) ) )
672   FORMAT(   T5,I2, 1P, ( T9, 8E9.1 ) )
      E N D
















      SUBROUTINE    Q Q E X    ( EWF, MAXST, IVEC, IEX, ISTN, ISTP )
C **********************************************************************
C *****  MODULE  Q Q E X  ( BASED  ON  BSYSME(#R02) / FEB 1982 )  ******
C *****   EXPECTATION  VALUE  OF  ( Q(N)*Q(N) )  &  ( Q(P)*Q(P) )  *****
C *****   IEX  =  1  :  ONLY  F**2                                 *****
C *****           2  :  ALL                                        *****
C **********************************************************************
      IMPLICIT  REAL*8(A-H,O-Z)
      PARAMETER   ( ICFP1=2907, ICFP2=6371, NEMAX=50, KLMAX=600 )
C **********************************************************************
      COMMON / QQEXCM / CHN, CHP, EXQ (NEMAX,NEMAX,3),
     &                  EXN (NEMAX,NEMAX,2), EXF (NEMAX,NEMAX)
C **********************************************************************
      COMMON / LAN   / NEIG, NVEC, EPS, LFILE, NCUT, BEFIX, ISYM,
     &                 IFILE, TRVASM
      COMMON / SPSTCM / ISTBS(300),NBD(300),LD(300),NSEN(300),NDL(300)
     1                 ,NBBAS(16),ISTBS2(300),LAD(300),LADBS(16)
      COMMON / CFP1CM / IDAU(ICFP1),L1(ICFP1),CFP1(ICFP1)
      COMMON / CFP2CM / IDAU2(ICFP2),L2(ICFP2),CFP2(ICFP2),LIM(ICFP2)
      COMMON / NPST / NDN(200),NDP(200),NPBS(200),
     &                NDT(20),NDTBS(20),NQBS4(24),NST
      COMMON / O    / ENERGY(KLMAX), JSTAT(KLMAX), FSTAT(KLMAX),
     &                KLEVEL, KLVL1
      COMMON / SR / SR(0:500)
      EQUIVALENCE  (SQ5,SR(5))
      COMMON / NBCNT / NDUPT, NDUPN, NDUPP, L, NNF2, NPF2
      COMMON / OUTCM / ICMW, IDBG, NPSTW, MATW, IWCF
      DIMENSION   EXQQNP(NEMAX,NEMAX,4)
 
      DIMENSION   ISTN(1), ISTP(1)
      DIMENSION   EWF(MAXST,1)
C ********************************************************************
      DATA    ISET / 0 /
C ********************************************************************
      NQBS( NT ) = NQBS4 ( NT + 4 )
      AND1(I) = IAND(1,I)
 
      IF(ISET.EQ.1)  GOTO  90
      SQH = DSQRT( 0.5D0 )
      C0N = -4 + 10*CHN*CHN*DRAC(4,4,4,4,4,0)
      C2N = -4 + 10*CHN*CHN*DRAC(4,4,4,4,4,4)
      C4N = -4 + 10*CHN*CHN*DRAC(4,4,4,4,4,8)
      C0P = -4 + 10*CHP*CHP*DRAC(4,4,4,4,4,0)
      C2P = -4 + 10*CHP*CHP*DRAC(4,4,4,4,4,4)
      C4P = -4 + 10*CHP*CHP*DRAC(4,4,4,4,4,8)
      VWN = 2 * SR(5)
      VWP = 2 * SR(5)
      VYN = 2 * SR(2) * CHN
      VYP = 2 * SR(2) * CHP
      VAN = ( 3*C4N + 4*C2N ) / 7
      VCN = ( C4N - C2N ) / 14
      VBN = ( C0N - VAN + 12*VCN ) * 0.1D0
      VAP = ( 3*C4P + 4*C2P ) / 7
      VCP = ( C4P - C2P ) / 14
      VBP = ( C0P - VAP + 12*VCP ) * 0.1D0
C*****
      VXV = CHN
      VVX = CHP
      VXX2 = CHN*CHP
C*****
      ISET = 1
C*****                                        *****
   90 CSTN = 5 * NNF2
      CSTP = 5 * NPF2
      SPN = ( 2 * NNF2 - 6 + CHN*CHN )
      SPP = ( 2 * NPF2 - 6 + CHP*CHP )
      IF( IVEC .GT. NEMAX )  IVEC = NEMAX
C********************************************************************
      DO 700  J = 1, IVEC
      DO 700  I = 1, IVEC
        EXF( I, J ) = 0
700   CONTINUE
      DO 702  J = 1, IVEC
      DO 702  I = 1, IVEC
        EXN( I, J, 1 ) = 0
        EXN( I, J, 2 ) = 0
        EXQ( I, J, 1 ) = 0
        EXQ( I, J, 2 ) = 0
        EXQ( I, J, 3 ) = 0
        EXQQNP( I, J, 1 ) = 0
        EXQQNP( I, J, 2 ) = 0
        EXQQNP( I, J, 3 ) = 0
        EXQQNP( I, J, 4 ) = 0
702   CONTINUE
      REWIND   I F I L E
      DO 230  NE1 = 1, IVEC
        READ( IFILE )  ( EWF( I, NE1 ), I = 1, NST )
230   CONTINUE
      REWIND   I F I L E
C********************************************************************
      DO 200  I = 1, NST
      IN  = ISTN( I )
      IP  = ISTP( I )
      LN  = LD  ( IN )
      LP  = LD  ( IP )
      LN2 = 2*LN
      LP2 = 2*LP
      NN  = NBD ( IN )
      DBN = NN
      NP  = NBD( IP )
      DBP = NP
      NT  = NN + NP
      NSN = NSEN( IN )
      SN  = NSN
      NSP = NSEN( IP )
      SP = NSP
      FLN = LN * ( LN + 1 )
      FLP = LP * ( LP + 1 )
C*****                                                      *****
C***          D I A G O N A L   P A R T S                     ***
C*****                                                      *****
      DO 708  NE2 = 1, IVEC
      DO 708  NE1 = 1, IVEC
        EXF( NE1, NE2 ) = EXF( NE1, NE2 )
     &                  + ( NNF2 - NN ) * ( NPF2 - NP )
     &                    * EWF(I,NE1) * EWF(I,NE2)
708   CONTINUE
      IF( IEX .EQ. 1 )  GOTO  95
 
      WQN = CSTN + DBN * ( SPN + 0.5D0 * VAN * ( DBN - 1 ) )
     &           + VBN * ( DBN - SN ) * ( DBN + SN + 3 )
     &           + VCN * ( FLN - 6 * DBN )
      WQP = CSTP + DBP * ( SPP + 0.5D0 * VAP * ( DBP - 1 ) )
     &           + VBP * ( DBP - SP ) * ( DBP + SP + 3 )
     &           + VCP * ( FLP - 6 * DBP )
      DO 710  NE2 = 1, IVEC
      DO 710  NE1 = 1, IVEC
      EXN (NE1,NE2,1) = EXN(NE1,NE2,1)
     1     + DBN * EWF(I,NE1) * EWF(I,NE2)
      EXN (NE1,NE2,2) = EXN(NE1,NE2,2)
     1     + DBP * EWF(I,NE1) * EWF(I,NE2)
      EXQ(NE1,NE2,1) = EXQ(NE1,NE2,1)
     &                 + WQN * EWF(I,NE1) * EWF(I,NE2)
      EXQ(NE1,NE2,2) = EXQ(NE1,NE2,2)
     &                 + WQP * EWF(I,NE1) * EWF(I,NE2)
  710 CONTINUE
C*****                                                      *****
C***          H Y   F O R   N E U T R O N                     ***
C*****                                                      *****
      IL = ISTBS2( IN )
      IR = ISTBS2( IN + 1 )
      IF( IL .EQ. IR )  GO  TO  10
      IL = IL + 1
      DO 202  IA = IL, IR
      IAX = IA
      IF( LIM( IA ) - 2 )  202, 12, 10
  202 CONTINUE
      GO  TO  10
   12 DO 203  IA = IAX, IR
      IA1 = IA
      IF( LIM( IA ) .NE. 2 )  GO  TO  13
  203 CONTINUE
      IA1 = IR + 1
   13 IR = IA1 - 1
      KL = NQBS( NT - 1 )
      KR = NQBS( NT )
      IF( KL .EQ. KR )  GO  TO  10
      KL1 = KL + 1
      DO 201  K = KL1, KR
      KZ = ISTN( K )
      IF( NBD( KZ ) .NE. ( NN - 1 ) )  GO  TO  201
      IF( LD( KZ ) .NE. LN )    GO  TO  201
      IF( ISTP( K ) .NE. IP )  GO  TO  201
      IA = IAX
      HYN = 0
      JA = ISTBS( KZ ) + 1
      JR = ISTBS( KZ + 1 )
   14 IB = IDAU2( IA )
   15 JB = IDAU( JA )
   16 IF( IB - JB )  17, 18, 19
   19 JA = JA + 1
      IF( JA .LE. JR )  GO  TO  15
      GO  TO  20
   17 IA = IA + 1
      IF( IA .GT. IR )  GO  TO  20
      IB = IDAU2( IA )
      GO  TO  16
   18 HYN = HYN + CFP2( IA ) * CFP1( JA )
      JA = JA + 1
      IA = IA + 1
      IF( JA .LE. JR .AND. IA .LE. IR )  GO  TO  14
   20 WQN = VYN * ( DBN - 1 ) * HYN
     2         * SQH * SR( NN ) * SR( NNF2 + 1 - NN )
      DO 720  NE2 = 1, IVEC
      DO 720  NE1 = 1, IVEC
      EXQ(NE1,NE2,1) = EXQ(NE1,NE2,1)
     &      + WQN * ( EWF(I,NE1)*EWF(K,NE2) + EWF(K,NE1)*EWF(I,NE2) )
  720 CONTINUE
  201 CONTINUE
C*****                                                      *****
C***          H Y   F O R   P R O T O N                       ***
C*****                                                      *****
   10 IL = ISTBS2( IP )
      IR = ISTBS2( IP + 1 )
      IF( IL .EQ. IR )  GO  TO  30
      IL = IL + 1
      DO 302  IA = IL, IR
      IAX = IA
      IF( LIM( IA ) - 2 )  302, 32, 30
  302 CONTINUE
      GO  TO  30
   32 DO 303  IA = IAX, IR
      IA1 = IA
      IF( LIM( IA ) .NE. 2 )  GO  TO  33
  303 CONTINUE
      IA1 = IR + 1
   33 IR = IA1 - 1
      KL = NQBS( NT - 1 )
      KR = NQBS( NT )
      IF( KL .EQ. KR )  GO  TO 30
      KL1 = KL + 1
      DO 301  K = KL1 , KR
      KZ = ISTP( K )
      IF( NBD( KZ ) .NE. ( NP - 1 ) )  GO  TO  301
      IF( LD( KZ ) .NE. LP )    GO  TO  301
      IF( ISTN( K ) .NE. IN )  GO  TO  301
      IA = IAX
      HYP = 0
      JA = ISTBS( KZ ) + 1
      JR = ISTBS( KZ + 1 )
   34 IB = IDAU2( IA )
   35 JB = IDAU( JA )
   36 IF( IB - JB )  37, 38, 39
   39 JA = JA + 1
      IF( JA .LE. JR )  GO  TO  35
      GO  TO  40
   37 IA = IA + 1
      IF( IA .GT. IR )  GO  TO  40
      IB = IDAU2( IA )
      GO  TO  36
   38 HYP = HYP + CFP2( IA ) * CFP1( JA )
      JA = JA + 1
      IA = IA + 1
      IF( JA .LE. JR .AND. IA .LE. IR )  GO  TO  34
   40 WQP = VYP * ( DBP - 1 ) * HYP
     2         * SQH * SR( NP ) * SR( NPF2 + 1 - NP )
      DO 730  NE2 = 1, IVEC
      DO 730  NE1 = 1, IVEC
      EXQ(NE1,NE2,2) = EXQ(NE1,NE2,2)
     &      + WQP * ( EWF(I,NE1)*EWF(K,NE2) + EWF(K,NE1)*EWF(I,NE2) )
  730 CONTINUE
  301 CONTINUE
C*****                                                      *****
C***          H W   F O R   N E U T R O N                     ***
C*****                                                      *****
   30 IF( NN .EQ. NSN )  GO  TO  50
      KL = NQBS( NT - 2 )
      KR = NQBS( NT - 1 )
      IF( KL .EQ. KR )  GO  TO  50
      IR = ISTBS2( IN + 1 )
      IL = ISTBS2( IN ) + 1
      DO 205  IA = IL, IR
      IF( LIM( IA ) .NE. 0 )  GO  TO  50
      IB = IDAU2( IA )
      IF( NSEN( IB ) .NE. NSN )  GO  TO  205
      KL1 = KL + 1
      DO 206  K = KL1, KR
      IF( IB .EQ. ISTN( K ) .AND. IP .EQ. ISTP( K ) )  GO  TO  52
  206 CONTINUE
      GO  TO  205
   52 WQN = VWN * 0.5D0 * CFP2( IA )
     2         * SR( NN ) * SR( NN - 1 ) * SR( NNF2 + 2 - NN )
     3         * SR( NNF2 + 1 - NN )
      DO 740  NE2 = 1, IVEC
      DO 740  NE1 = 1, IVEC
      EXQ(NE1,NE2,1) = EXQ(NE1,NE2,1)
     &      + WQN * ( EWF(I,NE1)*EWF(K,NE2) + EWF(K,NE1)*EWF(I,NE2) )
  740 CONTINUE
  205 CONTINUE
C*****                                                      *****
C***          H W   F O R   P R O T O N                       ***
C*****                                                      *****
   50 CONTINUE
      IF( NP .EQ. NSP )  GO  TO  60
      IR = ISTBS2( IP + 1 )
      IL = ISTBS2( IP ) + 1
      KL = NQBS( NT - 2 )
      KR = NQBS( NT - 1 )
      IF( KL .EQ. KR )  GO  TO  60
      DO 305  IA = IL, IR
      IF( LIM( IA ) .NE. 0 )  GO  TO  60
      IB = IDAU2( IA )
      IF( NSEN( IB ) .NE. NSP )  GO  TO  305
      KL1 = KL + 1
      DO 306  K = KL1, KR
      IF( IB .EQ. ISTP( K ) .AND. IN .EQ. ISTN( K ) )  GO  TO  62
  306 CONTINUE
      GO  TO  305
   62 WQP = VWP * 0.5D0 * CFP2( IA )
     2      * SR( NP ) * SR( NP - 1 ) * SR( NPF2 + 2 - NP )
     3      * SR( NPF2 + 1 - NP )
      DO 750  NE2 = 1, IVEC
      DO 750  NE1 = 1, IVEC
      EXQ(NE1,NE2,2) = EXQ(NE1,NE2,2)
     &      + WQP * ( EWF(I,NE1)*EWF(K,NE2) + EWF(K,NE1)*EWF(I,NE2) )
  750 CONTINUE
  305 CONTINUE
   60 CONTINUE
C*****                                                             *****
C***          H V V   F O R   N E U T R O N - P R O T O N            ***
C*****                                                             *****
      IF( NN .EQ. 0 .OR. NP .EQ. 0 )  GO  TO  150
      KL = NQBS( NT - 2 )
      KR = NQBS( NT - 1 ) + 1
      IF( KL .EQ. KR )  GO TO 100
      IL = ISTBS( IN ) + 1
      IR = ISTBS( IN + 1 )
      JL = ISTBS( IP ) + 1
      JR = ISTBS( IP + 1 )
      KL1 = KL + 1
      DO 210  K = KL1, KR
      KN = ISTN( K )
      IF( NBD( KN ) .NE. NN - 1 )  GO TO 210
      IF( IABS(LD(KN)-LN) .GT. 2 .OR. LD(KN)+LN .LT. 2 )  GO  TO  210
      KP = ISTP( K )
      IF( NBD( KP ) .NE. NP - 1 )  GO TO 210
      IF( IABS(LD(KP)-LP) .GT. 2 .OR. LD(KP)+LP .LT. 2 )  GO  TO  210
      DO 211  IX = IL, IR
      IF( IDAU( IX ) - KN )  211, 321, 102
  211 CONTINUE
  102 STOP 00102
  321 DO 212  JX = JL, JR
      IF( IDAU( JX ) - KP )  212, 103, 104
  212 CONTINUE
  104 STOP 00104
  103 HVV = SR( NN * ( LN2 + 1 ) ) * SR( NP * ( LP2 + 1 ) )
     1    * CFP1( IX ) * CFP1( JX )
     2    * RACAR ( LN, LP, LD(KN), LD(KP) )
     3    * SR( NNF2 + 1 - NN ) * SR( NPF2 + 1 - NP )
      IF( AND1( LD(KN) + LP + L ) .EQ. 1 ) HVV = - HVV
      DO 760  NE2 = 1, IVEC
      DO 760  NE1 = 1, IVEC
      EXQ(NE1,NE2,3) = EXQ(NE1,NE2,3) + HVV
     &     * ( EWF(I,NE1)*EWF(K,NE2) + EWF(K,NE1)*EWF(I,NE2) )
      EXQQNP(NE1,NE2,1) = EXQQNP(NE1,NE2,1) + HVV
     &     * ( EWF(I,NE1)*EWF(K,NE2) + EWF(K,NE1)*EWF(I,NE2) )
  760 CONTINUE
  210 CONTINUE
C*****                                                             *****
C***          H X V   F O R   N E U T R O N - P R O T O N            ***
C*****                                                             *****
  100 KL = NQBS( NT - 1 )
      KR = NQBS( NT )
      IF( KL .EQ. KR )  GO TO 73
      IL = ISTBS( IN ) + 1
      IR = ISTBS( IN + 1 )
      JL = ISTBS( IP ) + 1
      JR = ISTBS( IP + 1 )
      KL1 = KL + 1
      DO 214  K = KL1, KR
      KP = ISTP( K )
      IF( NBD(KP) .NE. NP - 1 )  GO TO 214
      IF( IABS(LD(KP)-LP) .GT. 2 .OR. LD(KP)+LP .LT. 2 )  GO  TO  214
      KN = ISTN( K )
      IF( NBD( KN ) .NE. NN )  GO TO 214
      IF( IABS(LD(KN)-LN) .GT. 2 .OR. LD(KN)+LN .LT. 2 )  GO  TO  214
      DO 215  JX = JL, JR
      IF( IDAU( JX ) - KP )  215, 70, 71
  215 CONTINUE
   71 STOP 00071
   70 HXV = DDMAT( KN, IN, 2 )
     1    * SR( NP * ( LP2 + 1 ) ) * CFP1( JX ) * VXV
     2    * RACAR ( LN, LP, LD(KN), LD(KP) )
     3    * SR( NPF2 + 1 - NP )
      IF( AND1( L + LP + LD(KN) ) .EQ. 1 )  HXV = - HXV
        DO 770  NE2 = 1, IVEC
        DO 770  NE1 = 1, IVEC
          EXQ(NE1,NE2,3) = EXQ(NE1,NE2,3) + HXV
     &           * ( EWF(I,NE1)*EWF(K,NE2) + EWF(K,NE1)*EWF(I,NE2) )
          EXQQNP(NE1,NE2,2) = EXQQNP(NE1,NE2,2) + HXV
     &           * ( EWF(I,NE1)*EWF(K,NE2) + EWF(K,NE1)*EWF(I,NE2) )
  770   CONTINUE
  214 CONTINUE
C*****                                                             *****
C***          H V X   F O R   N E U T R O N - P R O T O N            ***
C*****                                                             *****
   73 KL = NQBS( NT - 1 )
      KR = NQBS( NT )
      IF( KL .EQ. KR )  GO TO 93
      JL = ISTBS( IN ) + 1
      JR = ISTBS( IN + 1 )
      IF( JL .GT. JR )  GO  TO  93
      IL = ISTBS( IP ) + 1
      IR = ISTBS( IP + 1 )
      IF( IL .GT. IR )  GO  TO  93
      KL1 = KL + 1
      DO 314  K = KL1, KR
      KN = ISTN( K )
      IF( NBD(KN) .NE. NN - 1 )  GO TO 314
      IF( IABS(LD(KN)-LN) .GT. 2 .OR. LD(KN)+LN .LT. 2 )  GO  TO  314
      KP = ISTP( K )
      IF( NBD( KP ) .NE. NP )  GO TO 314
      IF( IABS(LD(KP)-LP) .GT. 2 .OR. LD(KP)+LP .LT. 2 )  GO  TO  314
      DO 315  JX = JL, JR
      IF( IDAU( JX ) - KN )  315, 390, 91
  315 CONTINUE
   91 STOP 00091
  390 HVX = DDMAT( KP, IP, 2 )
     1    * SR( NN * ( LN2 + 1 ) ) * CFP1( JX ) * VVX
     2    * RACAR ( LN, LP, LD(KN), LD(KP) )
     3    * SR( NNF2 + 1 - NN )
      IF( AND1( L + LP + LD(KN) ) .EQ. 1 )  HVX = - HVX
        DO 780  NE2 = 1, IVEC
        DO 780  NE1 = 1, IVEC
          EXQ(NE1,NE2,3) = EXQ(NE1,NE2,3) + HVX
     &          * ( EWF(I,NE1)*EWF(K,NE2) + EWF(K,NE1)*EWF(I,NE2) )
          EXQQNP(NE1,NE2,3) = EXQQNP(NE1,NE2,3) + HVX
     &          * ( EWF(I,NE1)*EWF(K,NE2) + EWF(K,NE1)*EWF(I,NE2) )
  780   CONTINUE
  314 CONTINUE
C*****                                                             *****
C***          H X X   F O R   N E U T R O N - P R O T O N            ***
C*****                                                             *****
   93 KL = NQBS( NT )
      IL = ISTBS( IN ) + 1
      IR = ISTBS( IN + 1 )
      JL = ISTBS( IP ) + 1
      JR = ISTBS( IP + 1 )
      M = 2
      KL1 = KL + 1
      DO 220  K = KL1, I
      KP = ISTP( K )
      IF( NBD(KP) .NE. NP )  GO TO 220
      IF( IABS(LD(KP)-LP) .GT. M .OR. LD(KP)+LP .LT. M )  GO  TO  220
      KN = ISTN( K )
      IF( NBD(KN) .NE. NN )  GO TO 220
      IF( IABS(LD(KN)-LN) .GT. M .OR. LD(KN)+LN .LT. M )  GO  TO  220
      HXX = DDMAT( KN, IN, M )
     1    * DDMAT( KP, IP, M ) * VXX2
     2    * DRACLM( LN, LP, LD(KN), LD(KP), L, M )
      IF(K.EQ.I) HXX = HXX / 2
      IF( AND1( ( L + LP + LD(KN) ) ) .EQ. 1 )  HXX = - HXX
        DO 790  NE2 = 1, IVEC
        DO 790  NE1 = 1, IVEC
          EXQ(NE1,NE2,3) = EXQ(NE1,NE2,3) + HXX
     &         * ( EWF(I,NE1)*EWF(K,NE2) + EWF(K,NE1)*EWF(I,NE2) )
          EXQQNP(NE1,NE2,4) = EXQQNP(NE1,NE2,4) + HXX
     &         * ( EWF(I,NE1)*EWF(K,NE2) + EWF(K,NE1)*EWF(I,NE2) )
  790   CONTINUE
  220 CONTINUE
 
95    KL = NQBS( NT )
      IL = ISTBS( IN ) + 1
      IR = ISTBS( IN + 1 )
      JL = ISTBS( IP ) + 1
      JR = ISTBS( IP + 1 )
      DO 400  M = 0, 4
        DO 420  K = KL+1, I
          KP = ISTP( K )
          IF( NBD(KP) .NE. NP )  GOTO 420
          IF( IABS(LD(KP)-LP) .GT. M .OR. LD(KP)+LP .LT. M )  GOTO  420
          KN = ISTN( K )
          IF( NBD(KN) .NE. NN )  GOTO 420
          IF( IABS(LD(KN)-LN) .GT. M .OR. LD(KN)+LN .LT. M )  GOTO  420
          HXX = DDMAT( KN, IN, M )
     1        * DDMAT( KP, IP, M )
     2        * DRACLM( LN, LP, LD(KN), LD(KP), L, M )
          IF(K.EQ.I) HXX = HXX / 2
          IF( AND1( ( L + LP + LD(KN) ) ) .EQ. 1 )  HXX = - HXX
          DO 430  NE2 = 1, IVEC
          DO 430  NE1 = 1, IVEC
            EXF (NE1,NE2) = EXF (NE1,NE2)
     &         + HXX * ( EWF(I,NE1)*EWF(K,NE2) + EWF(K,NE1)*EWF(I,NE2) )
C$$$$       EXF (NE1,NE2) = EXF (NE1,NE2)
C$$$$&         + HXX * ( EWF(I,NE1)*EWF(K,NE2) + EWF(K,NE1)*EWF(I,NE2) )
430       CONTINUE
420     CONTINUE
400   CONTINUE
C*****                                                             *****
C***          H A V   F O R   N E U T R O N - P R O T O N            ***
C*****                                                             *****
  150 IF( NP .LE. 0 )  GO  TO  200
      IL = ISTBS( IP ) + 1
      IR = ISTBS( IP + 1 )
      KL = NQBS( NT )
      IF( KL .EQ. I  )  GO  TO  200
      DO 260  K = KL+1, I
      KN = ISTN( K )
      IF( NBD( KN ) .NE. NN + 1 )  GO TO 260
      IF( IABS(LD(KN)-LN) .GT. 2 .OR. LD(KN)+LN .LT. 2 )  GO  TO  260
      KP = ISTP( K )
      IF( NBD( KP ) .NE. NP - 1 )  GO TO 260
      IF( IABS(LD(KP)-LP) .GT. 2 .OR. LD(KP)+LP .LT. 2 )  GO  TO  260
      DO 261  IX = IL, IR
      IF( IDAU( IX ) - KP )  261, 161, 162
  261 CONTINUE
  162 STOP 00162
  161 JL = ISTBS( KN ) + 1
      JR = ISTBS( KN + 1 )
      DO 262  JX = JL, JR
      IF( IDAU( JX ) - IN )  262, 163, 164
  262 CONTINUE
  164 STOP 00164
  163 LDN = LD(KN)
      LDN2 = LDN*2
      LDP = LD(KP)
      HAV = SR( NP*(NN+1) ) * SR( LP2 + 1 ) * SR( LDN2 + 1 )
     1    * CFP1( IX ) * CFP1( JX )
     2    * RACAR ( LN, LP, LDN, LDP )
     3    * SR( NPF2 + 1 - NP ) * SR( NNF2 - NN )
      IF( AND1( LN  +  LP   + L ) .EQ. 1 )  HAV = - HAV
        DO 795  NE2 = 1, IVEC
        DO 795  NE1 = 1, IVEC
          EXQ(NE1,NE2,3) = EXQ(NE1,NE2,3) + HAV
     &            * ( EWF(I,NE1)*EWF(K,NE2) + EWF(K,NE1)*EWF(I,NE2) )
          EXQQNP(NE1,NE2,1) = EXQQNP(NE1,NE2,1) + HAV
     &            * ( EWF(I,NE1)*EWF(K,NE2) + EWF(K,NE1)*EWF(I,NE2) )
          EXF(NE1,NE2)   = EXF(NE1,NE2)   + HAV
     &            * ( EWF(I,NE1)*EWF(K,NE2) + EWF(K,NE1)*EWF(I,NE2) )
795     CONTINUE
260   CONTINUE
C**********************************************************************
200   CONTINUE
 
      DO 440  NE1 = 1, IVEC
        EXF( NE1, NE1 ) = EXF( NE1, NE1 )
     &                  + 0.25 * ( ( NNF2 - NPF2 ) ** 2 )
     &                  + 0.5  * ( NNF2 + NPF2 )
440   CONTINUE
      KX = KLVL1
      DO 442  NE1 = 1, IVEC
        KX = KX + 1
        IF( KX .GT. KLMAX )  GOTO  442
        FSTAT( KX ) = EXF( NE1, NE1 )
442   CONTINUE
C**********************************************************************
      WRITE(6,630)  ( I, I = 1, IVEC )
      WRITE(6,640)
      DO 820  NE2 = 1, IVEC
        WRITE(6,632)  NE2, ( EXF(NE1,NE2), NE1=1,IVEC )
820   CONTINUE
      IF( IEX .LE. 1 )  RETURN
 
      WRITE(6,600)  ( I, I = 1, IVEC )
      WRITE(6,640)
      DO 800  NE2 = 1, IVEC
      WRITE(6,602)  NE2, ( EXN(NE1,NE2,1), NE1=1,IVEC )
      WRITE(6,604)       ( EXN(NE1,NE2,2), NE1=1,IVEC )
  800 CONTINUE
      WRITE(6,610)  ( I, I = 1, IVEC )
      WRITE(6,640)
      DO 810  NE2 = 1, IVEC
      WRITE(6,620)  NE2, ( EXQ(NE1,NE2,1), NE1=1,IVEC )
      WRITE(6,622)       ( EXQ(NE1,NE2,2), NE1=1,IVEC )
      WRITE(6,624)       ( EXQ(NE1,NE2,3), NE1=1,IVEC )
  810 CONTINUE
      WRITE(6,640)
      DO 830  NE2 = 1, IVEC
        WRITE(6,651)  NE2, ( EXQQNP(NE1,NE2,1), NE1=1,IVEC )
        WRITE(6,652)       ( EXQQNP(NE1,NE2,2), NE1=1,IVEC )
        WRITE(6,653)       ( EXQQNP(NE1,NE2,3), NE1=1,IVEC )
        WRITE(6,654)       ( EXQQNP(NE1,NE2,4), NE1=1,IVEC )
830   CONTINUE

      R E T U R N

C**********************************************************************
  600 FORMAT(//T7,'***  MATRIX  ELEMENT  OF  ND  ***'
     &  / T8,'ST#', (T20, 6(I2, 8X) ) )
  602 FORMAT(T9,I2,T13,'N', (T18,6(F8.4,2X)) )
  604 FORMAT(          T13,'P', (T18,6(F8.4,2X)) )
  610 FORMAT(//T7,'***  MATRIX  ELEMENT  OF  ( Q * Q )  ***'
     &  / T8,'ST#', (T20, 6(I2, 8X) ) )
  620 FORMAT(T9,I2,T13,'N-N', (T18,6(F8.4,2X)) )
  622 FORMAT(          T13,'P-P', (T18,6(F8.4,2X)) )
  624 FORMAT(          T13,'N-P', (T18,6(F8.4,2X)) )
630   FORMAT(//T7,'***  MATRIX  ELEMENT  OF  F**2  ***'
     &  / T8,'ST#', (T18, 6(I2, 8X) ) )
632   FORMAT(T9,I2, (T15,6(F8.4,2X)) )
640   FORMAT( T2,' ')
651   FORMAT(T9,I2,T13,'SD(N)*SD(P)', (T26,6(F8.4,2X)) )
652   FORMAT(      T13,'DD(N)*SD(P)', (T26,6(F8.4,2X)) )
653   FORMAT(      T13,'SD(N)*DD(P)', (T26,6(F8.4,2X)) )
654   FORMAT(      T13,'DD(N)*DD(P)', (T26,6(F8.4,2X)) )
      E N D








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







      SUBROUTINE     N P B S E T
      IMPLICIT  REAL*8(A-H,O-Z)
      PARAMETER  (NM=72)
      COMMON / OUTCM  / ICM(5)
C     COMMON / BCODCM / Q(2704)
      COMMON / BCODCM / Q(2*NM*NM)
 
      DO 210  I = 1, 5
  210 ICM( I ) = 0
      DO 230  I = 1, 2*NM*NM
  230 Q( I ) = 0
C     CALL   B C O D I N  ( 52 )
      CALL   B C O D I N
      RETURN
      E N D
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
