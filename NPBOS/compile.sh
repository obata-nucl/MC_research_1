gfortran cfpgen.f
./a.out

gfortran npcfpg.f
./a.out <<EOF
   11
EOF

gfortran racfl.f
./a.out <<EOF
    8   8     9
EOF


gfortran ddmefl.f
./a.out <<EOF
   11
EOF


gfortran npbos.f -o npbos

gfortran npbtrn.f -o npbtrn
