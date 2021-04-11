SYSTEM=$1

SLURM_DIR=~/projects/git/CircularizationDissipationConstraints/source/slurm

sed\
    -e 's%@@SYSTEM@@%'"$SYSTEM"'%g'\
    ${SLURM_DIR}/ganymede/template.slurm\
    >\
    ${SLURM_DIR}/ganymede/${SYSTEM}.slurm
