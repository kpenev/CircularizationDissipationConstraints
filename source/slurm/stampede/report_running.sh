SLURM_DIR=~/projects/git/CircularizationDissipationConstraints/source/slurm/stampede

for f in $(ls ${SLURM_DIR}/NGC*.slurm ${SLURM_DIR}/M35*.slurm ${SLURM_DIR}/*.slurm.disabled 2>/dev/null); do
    CLUSTER=$(echo $(basename $f)|sed -e 's%\(NGC[0-9]*\)_.*%\1%' -e 's%\(M35\)_.*%\1%')
    JOBNAME=$(grep '^#SBATCH -J' $f|awk '{print $3;}')

    if [ "${f#*.slurm}" == ".disabled" ] ; then
        JOB_INFO="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        DUMMY="$JOB_INFO"
    else
        JOB_INFO=$(
            squeue -n $JOBNAME -o '%.15i %.9P %.15j %.2t %.10M %R'\
            |\
            egrep 'NGC|M35'
        )
        DUMMY="........................................................................"
    fi

    echo $JOBNAME

    for SYS_ID in $(echo "$JOBNAME"|sed -e 's%NGC[0-9]*_\([0-9]*\)/\([0-9]*\)/\([0-9]*\)_.*$%\1 \2 \3%' -e 's%M35_\([0-9]*\)/\([0-9]*\)/\([0-9]*\)_.*$%\1 \2 \3%'); do
            H5FNAME=${CLUSTER}_${SYS_ID}_mcmc_powerlawlgQ_samples.h5
            test -e $H5FNAME || continue
            for((CHAIN_IND=0; 1; ++CHAIN_IND)); do 
                CHAIN=$(awk -v c=$CHAIN_IND 'BEGIN{printf("chain%05d\n", c);}')
                E_INIT=$(h5dump -a ${CHAIN}/initial_eccentricity $H5FNAME \
                    |grep '(0)'\
                    |awk '{print $NF;}'
                )
                if [ "$E_INIT" == "0.8" ]; then 
                        NSAMPLES=$(\
                            h5dump -a "${CHAIN}/iteration" $H5FNAME\
                            |\
                            grep '(0)'\
                            |\
                            awk '{print $2;}'\
                        )
                        echo "${CLUSTER}_${SYS_ID} ($NSAMPLES): $JOB_INFO"
                        JOB_INFO=$DUMMY
                        break;
                fi
            done
            echo ''
    done
done
