#!/bin/bash

CLUSTER=$1
SAMPLING_MODE=$2
HPC=$3
shift 3
TEMPLATE=template_tacc.slurm
QUEUE="normal"
if [ "$HPC" == "ganymede" ]; then
    TEMPLATE=template_ganymede.slurm
    EXPECTED_NSYS=1
    echo "Ganymede not fully implemented yet!"
    exit
elif [ "$HPC" == "stampede" ]; then
    HPC="stampede2"
    EXPECTED_NSYS=3
    QUEUE="skx-normal"
elif [ "$HPC" == "josh" ]; then
    TEMPLATE=template_tacc_josh.slurm
    EXPECTED_NSYS=8
    HPC="ls6"
else
    if [ "$HPC" != "ls6" ]; then
        echo "Unrecognized HPC cluster: ${HPC}"
        exit
    fi
    EXPECTED_NSYS=8
fi

SYSTEMS="$@"
if [ $# -ne "${EXPECTED_NSYS}" ]; then
    echo "Expected ${EXPECTED_NSYS} systems, got $#. Not generating script"
    exit
fi

echo "Generating $CLUSTER slurm script for $HPC"

if [ "$CLUSTER" == "W19" ]; then
    CONFIG_FNAME="W19"
    SAMPLER="sample_windemuth_et_al.py "
else
    CONFIG_FNAME="open_cluster"
    SAMPLER="sample_sb1.py ${CLUSTER}_"
fi

CONFIG_FNAME="${CONFIG_FNAME}_${SAMPLING_MODE}.cfg"

JOINED_SYSTEMS=""
for SYS in $SYSTEMS; do
    if [ "$JOINED_SYSTEMS" == "" ]; then
        JOINED_SYSTEMS="$SYS"
    else
        JOINED_SYSTEMS="${JOINED_SYSTEMS}_$SYS"
    fi
done

SLURM_DIR_1=/home1/08402/vortebo/codebase/CircularizationDissipationConstraints/source/slurm
SLURM_DIR_2=/home1/08402/vortebo/slurm

mkdir -p ${SLURM_DIR_2}/${HPC}/${CLUSTER}

sed\
    -e 's%@@CLUSTER@@%'"$CLUSTER"'%g'\
    -e 's%@@SYSTEM_LIST@@%'"$SYSTEMS"'%g'\
    -e 's%@@JOINED_SYSTEMS@@%'"${JOINED_SYSTEMS}"'%g'\
    -e 's%@@CONFIG_FNAME@@%'"${CONFIG_FNAME}"'%g'\
    -e 's%@@SAMPLER@@%'"${SAMPLER}"'%g'\
    -e 's%@@HPC@@%'"${HPC}"'%g'\
    -e 's%@@QUEUE@@%'"${QUEUE}"'%g'\
    ${SLURM_DIR_1}/${TEMPLATE}\
    >\
    ${SLURM_DIR_2}/${HPC}/${CLUSTER}/reduced_${JOINED_SYSTEMS}_${SAMPLING_MODE}.slurm
