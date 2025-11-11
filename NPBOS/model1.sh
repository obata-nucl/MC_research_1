#!/bin/bash

MASS=$1
Bn=$2
eps=$3
kappa=$4
chi_n=$5

./npbos <<EOF
 &N  NCUT=100,   IEX =1,
     LAUTO = 0, 2, 4, 6,
     NEIGA = 2, 1, 1, 1,
     NDUPTA= 8, 8, 8, 8, 8, 8, 8,
     IWCF= 2,  NPSTW= 0,
 &END
$MASS Sm
   $Bn   6
    0
 &INPT
      ED   = $eps,
      RKAP = $kappa,
      CHN  = $chi_n,
      CHP  = -0.5,

 &END
 
E
EOF

if [ ! -f "out1.dat" ]; then
    echo "Error: out1.dat not generated" >&2
    exit 0
fi