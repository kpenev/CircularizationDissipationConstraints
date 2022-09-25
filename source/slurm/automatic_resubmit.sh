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

for slurm in $(ls ~/projects/git/CircularizationDissipationConstraints/source/slurm/$HPC/$CLUSTER/*.slurm 2>/dev/null); do
	JOBNAME=$(grep '#SBATCH  *-J' $slurm|awk '{print $3;}')
	if [ "$(squeue -n $JOBNAME -h)" == "" ] ; then 
		sbatch "$slurm"
	else 
		echo "$JOBNAME still running"
	fi
done
