SLURM_DIR=~/projects/git/CircularizationDissipationConstraints/source/slurm/stampede

DUMMY="........................................................................"

for f in ${SLURM_DIR}/NGC*.slurm; do
    CLUSTER=$(echo $(basename $f)|sed -e 's%\(NGC[0-9]*\)_.*%\1%')
    CHAIN='chain00000'
    JOBNAME=$(grep '^#SBATCH -J' $f|awk '{print $3;}')

    JOB_INFO=$(
        squeue -n $JOBNAME -o '%.15i %.9P %.15j %.2t %.10M %R'\
        |\
        grep NGC
    )

    for SYS_ID in $(egrep '^for (WOCS|PKM) in [0-9]* [0-9]* [0-9]*; do' $f\
        | tr -d ';'\
        | awk '{print $4, $5, $6;}'\
    ); do
        NSAMPLES=$(\
            h5dump\
                -a "${CHAIN}/iteration"\
                ${CLUSTER}_${SYS_ID}_mcmc_powerlawlgQ_samples.h5\
            |\
            grep '(0)'\
            |\
            awk '{print $2;}'\
        )
        echo "${CLUSTER}_${SYS_ID} ($NSAMPLES): $JOB_INFO"
        JOB_INFO=$DUMMY
    done
    echo ''
done
