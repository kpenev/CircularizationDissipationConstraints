#!/bin/bash

CLUSTER=$1

HOSTNAME=$(hostname)
HOSTNAME=${HOSTNAME#login?.}

if [ "$HOSTNAME" == "ls6.tacc.utexas.edu" ]; then
	HPC='ls6'
elif [ "$HOSTNAME" == "stampede2.tacc.utexas.edu" ]; then 
	HPC='stampede'
elif [ "$HOSTNAME" == "ganymede.utdallas.edu" ]; then 
	HPC='ganymede'
else
	echo "Unrecognized host name: $HOSTNAME"
	exit 1
fi

OUTPUT_ROOT=/work/05392/kpenev/${HPC}/circularization/${CLUSTER}/sampling_output


for slurm in $(ls ~/projects/git/CircularizationDissipationConstraints/source/slurm/$HPC/$CLUSTER/*.slurm 2>/dev/null); do
	JOBNAME=$(grep '#SBATCH  *-J' $slurm|awk '{print $3;}')
	if [ "$(squeue -n $JOBNAME -h)" == "" ] ; then 
        for SYSNAME in $(echo $JOBNAME|tr '_' ' '); do 
            if [ "$SYSNAME" != "$CLUSTER" ] ; then
                OLD=${OUTPUT_ROOT}/${SYSNAME}
                NEW=${OLD}_$(date '+%Y%m%d')
                echo "Archiving  $OLD -> $NEW.tbz"
                echo mv ${OLD} ${NEW}
                echo tar -cjf ${NEW}.tbz ${NEW}
            fi
        done
		sbatch "$slurm"
	else 
		echo "$JOBNAME still running"
	fi
done
