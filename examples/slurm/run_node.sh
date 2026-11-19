#!/bin/bash -l
set -euo pipefail
: "${PACKAGE_ROOT:?Set PACKAGE_ROOT to a configured package checkout}"
: "${RUN_ROOT:?Set RUN_ROOT to a new output directory}"
: "${MASSIVE_STAR_GRID_ROOT:?Set the validated stellar-grid directory}"
: "${ECCENTRICITY_COEFFICIENTS:?Set the external coefficient SQLite path}"
SYSTEM_ID=$1
SYSTEM_ROOT="$RUN_ROOT/$SYSTEM_ID"
mkdir -p "$SYSTEM_ROOT"/{logs,cache,failed_solutions,solver_data}
module load gsl/2.8
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PACKAGE_ROOT:$PACKAGE_ROOT/source:$PACKAGE_ROOT/control:$PACKAGE_ROOT/general_purpose_python_modules:$PACKAGE_ROOT/poet/PythonPackage"
BOOST_RUNTIME_DIR=${BOOST_RUNTIME_DIR:-/work/dal205149/nesty_sampling_hpc/runtime_libs/boost-1.71}
export LD_LIBRARY_PATH="$BOOST_RUNTIME_DIR:$PACKAGE_ROOT/poet/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$SYSTEM_ROOT/cache" MPLCONFIGDIR="$SYSTEM_ROOT/cache/matplotlib"
export FAILED_SOLUTIONS_DIR="$SYSTEM_ROOT/failed_solutions"
TMP_ROOT=${TMP_ROOT:-/tmp}
mkdir -p "$TMP_ROOT"
export TMPDIR
TMPDIR=$(mktemp -d "$TMP_ROOT/circ-sampler.XXXXXXXX")
trap 'rmdir "$TMPDIR" 2>/dev/null || true' EXIT
export NESTY_DYNAMIC_NESTED=1 NESTY_DYNAMIC_MAXBATCH=200
export NESTY_POSTERIOR_CDF_STOP=1 NESTY_POSTERIOR_CDF_TOL=0.02
export NESTY_POSTERIOR_CDF_NREAL=128 NESTY_POSTERIOR_CDF_CONFIDENCE=0.95
export NESTY_CHECKPOINT_EVERY=0
: "${GAIA_SYSTEMS_CSV:?Set the external observation CSV path}"
export GAIA_SYSTEMS_CSV
cd "$PACKAGE_ROOT"
${PYTHON_BIN:-python} -u "$PACKAGE_ROOT/control/gaia_launcher.py" "$SYSTEM_ID" \
 --num-parallel-processes "${NWORKERS:-64}" --dynesty-nlive "${NLIVE:-64}" \
 --eccentricity-expansion-coefficients "$ECCENTRICITY_COEFFICIENTS" \
 --samples-fname "$SYSTEM_ROOT/${SYSTEM_ID}_nested_samples" \
 --logging-fname "$SYSTEM_ROOT/logs/{system}_{sampling}_{now}_{pid:d}.log" \
 --std-out-err-fname "$SYSTEM_ROOT/logs/{system}_{sampling}_{now}_{pid:d}.outerr" \
 --nn-data-dir "$SYSTEM_ROOT/solver_data"
