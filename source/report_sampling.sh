#!/bin/bash

for mode in sampling_output*; do
    echo "============== $mode ==============="
    for outcome in Success ERROR 'Calculating evolution failed'; do 
        echo ${outcome}: 
        count=$(grep "${outcome}" $mode/*.log|wc -l)
        if [ "$count" != "0" -a "$1" == "--detailed" ]; then
            egrep "lgQ_min|${outcome}" $(grep -l "${outcome}" $mode/*.log)|grep "${outcome}" -B 1|grep 'lgQ_min'|sort -g -k 3
        fi
        echo $count
    done
done
