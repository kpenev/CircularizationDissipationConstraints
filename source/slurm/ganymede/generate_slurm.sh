SYSTEM=$1
FLAVOR=$2

if [ "$FLAVOR" == "" ]; then
    FLAVOR="constQ"
fi

SLURM_DIR=~/projects/git/CircularizationDissipationConstraints/source/slurm

sed\
    -e 's%@@SYSTEM@@%'"$SYSTEM"'%g'\
    -e 's%@@FLAVOR@@%'"$FLAVOR"'%g'\
    ${SLURM_DIR}/ganymede/template.slurm\
    >\
    ${SLURM_DIR}/ganymede/${SYSTEM}_${FLAVOR}.slurm
