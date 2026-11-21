#!/bin/bash -l
#SBATCH --job-name=massive-dynamic
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
set -uo pipefail
: "${PACKAGE_ROOT:?Set PACKAGE_ROOT}"
: "${RUN_ROOT:?Set RUN_ROOT}"
: "${SYSTEMS_FILE:?Set an external one-ID-per-line target file}"
mapfile -t SYSTEMS < "$SYSTEMS_FILE"
[[ ${#SYSTEMS[@]} -eq ${SLURM_JOB_NUM_NODES} ]] || exit 2
pids=()
for SYSTEM_ID in "${SYSTEMS[@]}"; do
 mkdir -p "$RUN_ROOT/$SYSTEM_ID"
 srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=0 --cpu-bind=cores \
  --output="$RUN_ROOT/$SYSTEM_ID/launcher-%j.log" \
  bash -l "$PACKAGE_ROOT/examples/slurm/run_node.sh" "$SYSTEM_ID" &
 pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
 wait "$pid" || status=1
done
exit "$status"
