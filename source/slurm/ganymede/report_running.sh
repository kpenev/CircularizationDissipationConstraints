SLURM_DIR=~/projects/git/CircularizationDissipationConstraints/source/slurm/ganymede
for f in ${SLURM_DIR}/NGC*.slurm; do 
    sys=$(basename ${f%.slurm})
    if [ "$sys" == "NGC188_4618" ]; then
        chain='chain00010'
    else
        chain='chain00000'
    fi
    nsamples=$(\
        h5dump -a "${chain}/iteration" ${sys}_mcmc_samples.h5\
        |\
        grep '(0)'\
        |\
        awk '{print $2;}'\
    )
    echo -n "$(awk -v sys="$sys" -v nsamples="$nsamples" 'BEGIN{printf("%23s (%3d):", sys, nsamples);}') $(squeue -n $sys -o '%.8i %.9P %.18j %.2t %.10M %R'|grep NGC)" 
    if [ \
            "$sys" == "NGC188_5463" \
            -o \
            "$sys" == "NGC188_5601" \
            -o \
            "$sys" == "NGC188_4618" \
            -o \
            "$sys" == "NGC188_4999" \
            -o \
            "$sys" == "NGC188_8775"\
            -o \
            "$sys" == "NGC188_4904"\
            -o \
            "$sys" == "NGC188_5733"\
    ]; then
        echo "-----------------------------------------------------------------"
    else
        echo ""
    fi
done
