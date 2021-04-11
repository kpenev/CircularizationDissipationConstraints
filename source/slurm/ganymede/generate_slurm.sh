SYSTEM=$1

sed\
    -e 's%@@SYSTEM@@%'"$SYSTEM"'%g'\
    slurm/ganymede/template.slurm\
    >\
    ~/projects/git/CircularizationDissipationConstraints/source/slurm/ganymede/test_${SYSTEM}.slurm
