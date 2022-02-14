SLURM_DIR=~/projects/git/CircularizationDissipationConstraints/source/slurm/ganymede
for f in $(ls ${SLURM_DIR}/*.slurm ${SLURM_DIR}/*.slurm.disabled 2>/dev/null) ; do 
    SYS=$(basename ${f%.slurm})
    JOBNAME=$(grep '^#SBATCH -J' $f|awk '{print $3;}')
    H5FNAME=~/${SYS}_mcmc_samples.h5

    test -e $H5FNAME || continue

    if [ "${f#*.slurm}" == ".disabled" ] ; then
        JOB_INFO="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    else
        JOB_INFO=$(
            squeue -n $JOBNAME -o '%.15i %.9P %.15j %.2t %.10M %R'\
            |\
            egrep 'NGC|M35'
        )
    fi

    for((CHAIN_IND=0; 1; ++CHAIN_IND)); do 
        CHAIN=$(awk -v c=$CHAIN_IND 'BEGIN{printf("chain%05d\n", c);}')
        E_INIT=$(h5dump -a ${CHAIN}/initial_eccentricity $H5FNAME 2>/dev/null\
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
                echo "${SYS} ($NSAMPLES): $JOB_INFO"
                break;
        fi
        if [ "$E_INIT" == "" ]; then
            NSAMPLES=0
            echo "${SYS} ($NSAMPLES): $JOB_INFO"
            break;
        fi
    done
done
