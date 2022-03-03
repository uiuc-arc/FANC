#!/bin/bash

cd ../ELINA
# ./configure -use-deeppoly -use-gurobi -use-fconv && make && sudo make install
./configure -use-deeppoly -use-gurobi -use-fconv --mpfr-prefix ~/usr/local/ --cdd-prefix /home/sugare2/usr/local/include/cddlib/ --gmp-prefix /home/sugare2/usr/local/include/ && make && make install
cd ../tf_verify
