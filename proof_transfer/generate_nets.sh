#!/bin/bash

shopt -s expand_aliases
source ~/.bashrc

python3 --version

declare -a app=("float16" "quant16" "quant8" "prune")
declare -a mdl=("fcn" "conv")
declare -a ds=("mnist" "cifar")

## now loop through the above array
for app_i in "${app[@]}" 
do
    for mdl_i in "${mdl[@]}" 
    do
        for ds_i in "${ds[@]}" 
        do
            python3 approximate.py --approx_type $app_i --model $mdl_i --dataset $ds_i
        done
    done 
done

# python3 approximate.py --approx_type quant8 --model conv --dataset mnist