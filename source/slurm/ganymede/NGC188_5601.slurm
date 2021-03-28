#!/bin/bash

#SBATCH -J NGC188_5601
#SBATCH -n 16 #Total number of tasks to run at one time
#SBATCH -N 1  #Number of nodes on which to distribute the tasks evenly
#SBATCH -o /home/kxp174430/scratch/open_cluster_circularization/sampling_output/NGC188_5601_slurm.out
#SBATCH --mail-user=kxp174430@utdallas.edu
#SBATCH -t 4-00:00:00

module load boost

export PYTHONPATH=/home/kxp174430/projects/git/poet/PythonPackage:/home/kxp174430/projects/git/CircularizationDissipationConstraints/source:/home/kxp174430/projects/git/general_purpose_python_modules
export PATH=$PATH:/home/kxp174430/bin

for d in /home/kxp174430/projects/git/poet/build/libs/*/shared/release; do
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(readlink -e $d)
    export LIBRARY_PATH=$LIBRARY_PATH:$(readlink -e $d)
done

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kxp174430/lib
export LIBRARY_PATH=$LIBRARY_PATH:/home/kxp174430/lib

SOURCE_DIR=/home/kxp174430/projects/git/CircularizationDissipationConstraints/source
OUTPUT_ROOT=/home/kxp174430/scratch/open_cluster_circularization
OUTPUT_DIR=${OUTPUT_ROOT}/sampling_output

python3 \
    ${SOURCE_DIR}/bayesian/sample_sb1.py NGC188_5601 \
    -c ${SOURCE_DIR}/sampling_config/open_cluster_constQ_nolock.cfg\
    --std-out-err-fname ${OUTPUT_DIR}'/%(system)s_%(now)s_%(pid)d.outerr'\
    --logging-fname ${OUTPUT_DIR}'/%(system)s_%(now)s_%(pid)d.log'\
    --samples-fname '/home/kxp174430/%(system)s_%(sampling)s_samples'\
    >\
    ${OUTPUT_DIR}/sample_NGC188_5601_1.outerr\
    2>&1
