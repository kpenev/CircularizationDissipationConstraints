#!/bin/bash -l
#SBATCH --job-name=zams-spin-smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/work/dal205149/massive_star_circularization/logs/spin-smoke-%j.log
module load gsl/2.8
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for PACKAGE_ROOT in /work/dal205149/massive_star_circularization /work/dal205149/gaia_massive_star_circularization; do
 cd "$PACKAGE_ROOT" || exit 1
 export PYTHONPATH="$PWD:$PWD/source:$PWD/control:$PWD/general_purpose_python_modules:$PWD/poet/PythonPackage"
 export LD_LIBRARY_PATH="/work/dal205149/nesty_sampling_hpc/runtime_libs/boost-1.71:$PWD/poet/lib:${LD_LIBRARY_PATH:-}"
 export XDG_CACHE_HOME="$PWD/cache" MPLCONFIGDIR="$PWD/cache/matplotlib" TMPDIR="$PWD/tmp" FAILED_SOLUTIONS_DIR="$PWD/failed_solutions"
 timeout 240 /home/dal205149/.conda/envs/myenv/bin/python -u control/smoke_zams_spin.py
done
