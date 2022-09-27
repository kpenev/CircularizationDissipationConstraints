#!/bin/bash -l

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

TAR_LIST=""
for slurm in $(ls ~/projects/git/CircularizationDissipationConstraints/source/slurm/$HPC/$CLUSTER/*.slurm 2>/dev/null); do
    JOBNAME=$(grep '#SBATCH  *-J' $slurm|awk '{print $3;}')
    if [ "$(squeue -n $JOBNAME -h)" == "" ] ; then 
        for SYSNAME in $(echo $JOBNAME|tr '_' ' '); do 
            if [ "$SYSNAME" != "$CLUSTER" ] ; then
                OLD=${OUTPUT_ROOT}/${SYSNAME}
                NEW=${OUTPUT_ROOT}/archived/${SYSNAME}_$(date '+%Y%m%d')
                echo "Moving $OLD -> $NEW"
                test -e "${NEW}" && (echo "${NEW} already exists!"; exit)
                test -e "${NEW}.tbz" && (echo "${NEW}.tbz already exists!"; exit)
                mv "${OLD}" "${NEW}"
                TAR_LIST="${TAR_LIST}${NEW} "
            fi
        done
        sbatch "$slurm"
    else 
        echo "$JOBNAME still running"
    fi
done

for TO_TAR in ${TAR_LIST}; do
    echo "Archiving ${TO_TAR} -> ${TO_TAR}.tbz"
    tar -cjf ${TO_TAR}.tbz ${TO_TAR} --remove-files
done
