#!/bin/bash

CLUSTER=$1
SAMPLING_MODE=$2
shift 2
SYSTEMS="$@"
if [ $# -ne 8 ]; then
    echo "Expected 8 systems, got $#. Not generating script"
    exit
fi

echo "Generating $CLUSTER slurm script for LS6"

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

SLURM_DIR=~/projects/git/CircularizationDissipationConstraints/source/slurm
sed\
    -e 's%@@CLUSTER@@%'"$CLUSTER"'%g'\
    -e 's%@@SYSTEM_LIST@@%'"$SYSTEMS"'%g'\
    -e 's%@@JOINED_SYSTEMS@@%'"${JOINED_SYSTEMS}"'%g'\
    -e 's%@@CONFIG_FNAME@@%'"${CONFIG_FNAME}"'%g'\
    -e 's%@@SAMPLER@@%'"${SAMPLER}"'%g'\
    ${SLURM_DIR}/ls6/template_powerlaw.slurm\
    >\
    ${SLURM_DIR}/ls6/${CLUSTER}_${JOINED_SYSTEMS}_powerlaw.slurm
