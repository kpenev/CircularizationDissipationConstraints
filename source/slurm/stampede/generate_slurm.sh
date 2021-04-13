#!/bin/bash

CLUSTER=$1
SYS1=$2
SYS2=$3
SYS3=$4

FNAME_SUB="${CLUSTER}_${SYS1}_${SYS2}_${SYS3}"


SLURM_DIR=~/projects/git/CircularizationDissipationConstraints/source/slurm
sed\
    -e 's%@@CLUSTER@@%'"$CLUSTER"'%g'\
    -e 's%@@SYS1@@%'"$SYS1"'%g'\
    -e 's%@@SYS2@@%'"$SYS2"'%g'\
    -e 's%@@SYS3@@%'"$SYS3"'%g'\
    ${SLURM_DIR}/stampede/template_powerlaw.slurm\
    >\
    ${SLURM_DIR}/stampede/${CLUSTER}_${SYS1}_${SYS2}_${SYS3}_powerlaw.slurm
