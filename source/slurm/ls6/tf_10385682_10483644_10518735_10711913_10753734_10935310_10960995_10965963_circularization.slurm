#!/bin/bash

#SBATCH -J W19_10385682_10483644_10518735_10711913_10753734_10935310_10960995_10965963
#SBATCH -N 1  #Number of nodes on which to distribute the tasks evenly
#SBATCH -n 1 #Total number of tasks to run at one time
#SBATCH -o /scratch/08402/vortebo/ls6/output/W19/sampling_output/10385682_10483644_10518735_10711913_10753734_10935310_10960995_10965963_slurm.out
#SBATCH --mail-user=jas180011@utdallas.edu
#SBATCH --mail-type=all
#SBATCH -t 48:00:00
#SBATCH -p normal

trainsystem() {
    systemdir=/scratch/08402/vortebo/ls6/output/W19/nn_data
    systemname=$1

    cd /home1/08402/vortebo/codebase/CircularizationDissipationConstraints/source/bayesian
    python3 ml_clean.py ${systemname} ${systemdir}

    cd /home1/08402/vortebo/
    python3 trainsystem.py ${systemname} ${systemdir}
}

run_system() {
    source_dir=/home1/08402/vortebo/codebase/CircularizationDissipationConstraints/source
    config_dir=/home1/08402/vortebo/system_configs
    output_root=/scratch/08402/vortebo/ls6/output/W19
    output_dir=${output_root}/sampling_output
    sysid=$1

    mkdir -p ${output_dir}/${sysid}
    mkdir -p ${output_root}/samples

    cd /home1/08402/vortebo/

    for (( i=0; i< 10; i++ )); do
        python3 \
            ${source_dir}/bayesian/sample_windemuth_et_al.py ${sysid} \
            -c ${config_dir}/'windemuth_template.cfg'\
            --track-final-eccentricity\
            >\
            ${output_dir}/sample_W19_${sysid}.outerr\
            2>&1
        sleep 60
    done
}

source /home1/08402/vortebo/module_load_order.txt

export PYTHONPATH=$PYTHONPATH:.:$HOME/codebase/poet/scripts:$HOME/codebase/poet/PythonPackage:$HOME/codebase/general_purpose_python_modules:$HOME/codebase/CircularizationDissipationConstraints/data:$HOME/codebase:$HOME/codebase/CircularizationDissipationConstraints/source:$HOME/codebase/Initial-Condition-Solver-for-Tidal-Evolution-using-Probabilistic-Neural-Network
export PATH=$PATH:/home1/08402/vortebo/bin

export PYTHONUNBUFFERED=1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home1/08402/vortebo/lib
export LIBRARY_PATH=$LIBRARY_PATH:/home1/08402/vortebo/lib

for SYSID in 10385682 10483644 10518735 10711913 10753734 10935310 10960995 10965963; do 
    trainsystem $SYSID
done

for SYSID in 10385682 10483644 10518735 10711913 10753734 10935310 10960995 10965963; do 
    run_system $SYSID &
done

report_mem.sh 9

wait
