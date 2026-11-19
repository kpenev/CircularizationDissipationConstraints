#!/bin/bash

CLUSTER=$1

HOSTNAME=$(hostname)
HOSTNAME=${HOSTNAME#login?.}

if [ "$HOSTNAME" == "ls6.tacc.utexas.edu" ]; then
    HPC='ls6'
elif [ "$HOSTNAME" == "stampede2.tacc.utexas.edu" ]; then 
    HPC='stampede2'
elif [ "$HOSTNAME" == "ganymede.utdallas.edu" ]; then 
    HPC='ganymede'
else
    echo "Unrecognized host name: $HOSTNAME"
    exit 1
fi

OUTPUT_ROOT=/work/05392/kpenev/${HPC}/circularization/${CLUSTER}/sampling_output

for sysname in ${OUTPUT_ROOT}/[0-9]*[0-9]; do 
    badlist=$(
    for f in $(grep -L '[Ss]uccess' $sysname/calculate_*.log); do \
        test -s $f\
        && \
        (
            tail -n 6 $f\
                |head -n 1\
                |grep -q '^DEBUG .* orbital_evolution.initial_condition_solver: Evolving:' \
                || echo $f
        )
    done
    )
    leftover=$(
        for f in $badlist ; do 
            tail -n 3 $f | head -n 1 | grep -q 'Invalid parameter values encountered: Stellar metallicity: <Quantity .*> is outside the range supported by the stelalr evolution interpolator: -1.014 - 0.537' || echo $f
        done
    )
    echo '======================================='
    echo $sysname
    if [ "$leftover" != "" ] ; then 
        ls $leftover
    fi
done 

