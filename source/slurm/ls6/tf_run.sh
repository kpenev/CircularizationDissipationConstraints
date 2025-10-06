echo "Backing up ML data."
cd /work/08402/vortebo/ls6/
cd output/W19/nn_data
#######################
# rsync command goes here instead, rysnc poet_output over, then copy it as below (so change directories)
# ALSO rsync the most recent backup, because that is how it works? that's the one that gets compared against, because it becomes third, behind the two backups we make right here
rsync -rc $(dir --format=single-column|grep poet_output|tail -n 2) /scratch/08402/vortebo/ls6/output/W19/nn_data/
cd /scratch/08402/vortebo/ls6/output/W19/nn_data
#######################
cp -r poet_output $(date +%Y%m%d-%H%M%S)_poet_output_backup
sleep 2
cp -r poet_output $(date +%Y%m%d-%H%M%S)_poet_output_backup
echo "Backing up samples data."
#cd ..
cd /work/08402/vortebo/ls6/output/W19/
#######################
# again, rsync the thing first, then cd to scratch and make the copy
rsync -rc samples /scratch/08402/vortebo/ls6/output/W19/
cd /scratch/08402/vortebo/ls6/output/W19/
#######################
cp -r samples $(date +%Y%m%d-%H%M%S)_samples_backup
cd /home1/08402/vortebo/
cd slurm/ls6/W19
echo "Backups complete. Submitting jobs."
sbatch -A AST22013 tf_10385682_10483644_10518735_10711913_10753734_10935310_10960995_10965963_circularization.slurm
sbatch -A AST22013 tf_10992733_11071207_11200773_11228612_11232745_11234677_11252617_11391181_circularization.slurm
sbatch -A AST22013 tf_11403216_11499757_11616200_11619964_11704044_12004679_12316447_circularization.slurm
sbatch -A AST22013 tf_2437452_2445134_3003991_3241344_3348093_3427776_3439031_3834364_circularization.slurm
sbatch -A AST22013 tf_3973504_4276114_4285087_4346875_4352168_4380283_4579321_4678171_circularization.slurm
sbatch -A AST22013 tf_4753988_4815612_4839180_4908495_4947726_4948863_5022440_5039441_circularization.slurm
sbatch -A AST22013 tf_5181455_5263802_5288543_5359678_5393558_5622250_5652260_5731312_circularization.slurm
sbatch -A AST22013 tf_5781192_5802470_5871918_6029130_6131659_6185717_6227560_6301030_circularization.slurm
sbatch -A AST22013 tf_6312521_6359798_6521542_6522750_6525196_6546508_6594972_6610219_circularization.slurm
sbatch -A AST22013 tf_6697716_6927629_6949550_6962018_7021177_7025851_7118545_7125636_circularization.slurm
sbatch -A AST22013 tf_7128918_7129465_7200102_7257373_7362852_7369523_7376500_7377033_circularization.slurm
sbatch -A AST22013 tf_7597703_7691527_7732791_7798259_7846730_7960547_7970629_7987749_circularization.slurm
sbatch -A AST22013 tf_8111622_8229048_8302455_8356054_8364119_8381592_8414159_8543278_circularization.slurm
sbatch -A AST22013 tf_8580438_8618226_8621353_8746310_8957954_8984706_9001468_9025914_circularization.slurm
sbatch -A AST22013 tf_9110346_9119652_9353182_9532123_9656543_9665503_9715925_9762519_circularization.slurm
sbatch -A AST22013 tf_9775253_9881258_9892471_9965206_9971475_10031409_10091257_10268903_circularization.slurm
echo "Jobs submitted. Have a great day!"
