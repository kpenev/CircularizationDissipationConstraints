#!/usr/bin/env python3
"""Run and plot a constant-Q' POET scan for one Windemuth et al. binary.

The default system is KIC 3348093.  Both of its component-mass posteriors are
inside the 0.4--1.2 Msun range covered by the local POET interpolator.  For
each requested lgQ and initial eccentricity this script:

1. fixes the stellar/system parameters to the medians of the W19 samples;
2. solves for the initial orbital period that reproduces the observed period
   at the sampled median age;
3. retains the complete period(age) and eccentricity(age) trajectories; and
4. evaluates the W19 eccentricity-sample KDE at the predicted final
   eccentricity.

The endpoint likelihoods are marginalized over the supplied initial-
eccentricity grid and normalized over the lgQ grid using a uniform prior.
This is a useful one-system diagnostic, but it is not a replacement for the
full MCMC, which also samples the correlated stellar/system parameters.  If a
non-empty emcee HDF5 file containing an ``lgQ_min`` blob is supplied with
``--mcmc-file``, its post-burn-in samples are used for the title and the lgQ
panel instead of the diagnostic grid weights.

Outputs are a PNG figure, a per-trial summary CSV, a CSV containing every
plotted trajectory, an explicit lgQ-bounds JSON file, and a JSON run
configuration.

Passing ``--mass-eccentricity-grid`` instead runs an independent 3-D grid in
primary mass, secondary mass, and initial eccentricity.  That mode uses a
bounded process pool around the real CPU POET library, saves one atomic
trajectory checkpoint per point, and combines the endpoints into CSV and NPZ
files.  POET itself has no CUDA evolution backend; the separate CUDA examples
in this workspace implement a reduced model and are not used here.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parents[2]
WORKSPACE_DIR = SCRIPT_PATH.parents[3]
POET_DIR = WORKSPACE_DIR / "poet"
BOOTSTRAP_FLAG = "_POET_W19_LGQ_SCAN_BOOTSTRAPPED"
PHASE_LAG_TO_QPRIME = 15.0 / (16.0 * math.pi)


def _bootstrap_poet_runtime() -> None:
    """Restart once with locally compiled POET libraries on the loader path."""

    if os.environ.get(BOOTSTRAP_FLAG) == "1":
        return

    library_dirs = sorted(POET_DIR.glob("build/libs/*/shared/release"))
    if not library_dirs:
        raise RuntimeError(
            f"No compiled POET libraries found below {POET_DIR / 'build/libs'}"
        )

    env = os.environ.copy()
    entries = [str(directory) for directory in library_dirs]
    if env.get("LD_LIBRARY_PATH"):
        entries.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(entries)
    env.setdefault("MPLCONFIGDIR", "/tmp/poet_w19_matplotlib_cache")
    env[BOOTSTRAP_FLAG] = "1"
    os.execve(
        sys.executable,
        [sys.executable, str(SCRIPT_PATH), *sys.argv[1:]],
        env,
    )


if __name__ == "__main__":
    _bootstrap_poet_runtime()
sys.path.insert(0, str(POET_DIR / "PythonPackage"))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "general_purpose_python_modules"))
sys.path.insert(0, str(PROJECT_DIR / "source"))

import h5py  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from scipy.stats import norm as normal_distribution  # noqa: E402

from basic_utils import calc_semimajor  # noqa: E402
from orbital_evolution.binary import Binary  # noqa: E402
from orbital_evolution.evolve_interface import library as evolution_library  # noqa: E402
from orbital_evolution.planet_interface import LockedPlanet  # noqa: E402
from orbital_evolution.star_interface import EvolvingStar  # noqa: E402
from stellar_evolution.library_interface import (  # noqa: E402
    MESAInterpolator,
    library as stellar_evolution_library,
)
from stellar_evolution.manager import StellarEvolutionManager  # noqa: E402


W19_COLUMNS = [
    "Mtot",
    "Mratio",
    "z",
    "tau",
    "dist",
    "E(B-V)",
    "P",
    "tpe",
    "esinw",
    "ecosw",
    "b",
    "q11",
    "q12",
    "q21",
    "q22",
    "SysLCerror",
    "SysSEDerror",
    "E(B-V)unc",
    "distunc",
]


def first_existing_file(candidates: list[Path], description: str) -> Path:
    """Return the first non-empty file in candidates."""

    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {description}. Tried:\n  {tried}")


def find_w19_data_dir(explicit_path: str | None) -> Path:
    """Locate the directory containing the full W19 thinned chains."""

    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    if os.environ.get("W19_SAMPLES_DIR"):
        candidates.append(Path(os.environ["W19_SAMPLES_DIR"]).expanduser())
    candidates.extend(
        [
            Path(
                "/home/kpenev/projects/git/CircularizationDissipationConstraints/"
                "data/windemuth_et_al_19_samples"
            ),
            PROJECT_DIR / "data" / "windemuth_et_al_19_samples",
        ]
    )
    for candidate in candidates:
        if (
            candidate.is_dir()
            and (candidate / "paper_stellar.posteriors").is_file()
            and any(candidate.glob("*.npz"))
        ):
            return candidate.resolve()
    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find the W19 sample directory. Pass --w19-data-dir or set "
        f"W19_SAMPLES_DIR. Tried:\n  {tried}"
    )


def find_eccentricity_database(explicit_path: str | None) -> Path:
    """Locate a non-empty POET eccentricity-expansion database."""

    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    if os.environ.get("POET_ECCENTRICITY_DB"):
        candidates.append(Path(os.environ["POET_ECCENTRICITY_DB"]).expanduser())
    candidates.extend(
        [
            Path("/home/vortebo/eccentricity_expansion_coef_O400.sqlite"),
            POET_DIR / "eccentricity_expansion_coef_O400.sqlite",
            WORKSPACE_DIR / "eccentricity_expansion_coef_O400.sqlite",
        ]
    )
    return first_existing_file(candidates, "a non-empty eccentricity database")


def load_w19_samples(data_dir: Path, kic: int) -> dict[str, np.ndarray]:
    """Load a W19 thinned chain and add useful derived quantities."""

    sample_path = data_dir / f"{kic}.npz"
    if not sample_path.is_file():
        raise FileNotFoundError(f"No W19 chain for KIC {kic}: {sample_path}")
    with np.load(sample_path) as archive:
        raw = np.asarray(archive["thinned_chain"], dtype=float)

    columns = list(W19_COLUMNS)
    if raw.shape[1] == 18:
        del columns[17]
    if raw.ndim != 2 or raw.shape[1] != len(columns):
        raise ValueError(
            f"Unexpected W19 chain shape {raw.shape}; expected 18 or 19 columns"
        )

    result = {name: raw[:, index] for index, name in enumerate(columns)}
    result["primary_mass"] = result["Mtot"] / (1.0 + result["Mratio"])
    result["secondary_mass"] = result["primary_mass"] * result["Mratio"]
    result["age_gyr"] = 10.0 ** (result["tau"] - 9.0)
    result["eccentricity"] = np.hypot(result["esinw"], result["ecosw"])
    return result


def representative_system(samples: dict[str, np.ndarray]) -> dict[str, float]:
    """Return the independently sampled posterior medians used in this scan."""

    z_median = float(np.median(samples["z"]))
    return {
        "primary_mass": float(np.median(samples["primary_mass"])),
        "secondary_mass": float(np.median(samples["secondary_mass"])),
        "z": z_median,
        "feh": float(stellar_evolution_library.feh_from_z(z_median)),
        "age_gyr": float(np.median(samples["age_gyr"])),
        "orbital_period_day": float(np.median(samples["P"])),
        "eccentricity": float(np.median(samples["eccentricity"])),
    }


GRID_LOG_PATTERN = re.compile(
    r"^lgQ=(?P<lgq>[-+0-9.eE]+), initial e=(?P<ecc>[-+0-9.eE]+)\s*$"
)
SOLVED_LOG_PATTERN = re.compile(
    r"^\s*solved P_i=(?P<period>[-+0-9.eE]+) d;"
)


def trial_checkpoint_signature(
    *,
    args: argparse.Namespace,
    system: dict[str, float],
    data_dir: Path,
    eccentricity_database: Path,
) -> str:
    """Identify all inputs which affect a saved trial or its likelihood."""

    parameter_names = (
        "disk_dissipation_age",
        "disk_lock_period",
        "wind_strength",
        "wind_saturation",
        "core_envelope_coupling_timescale",
        "max_time_step",
        "precision",
        "eccentricity_precision",
        "timeout",
        "output_samples",
        "period_tolerance",
        "period_search_factor",
        "min_initial_period",
        "max_initial_period",
        "max_bracket_steps",
        "max_root_iterations",
    )
    payload = {
        "checkpoint_format": 1,
        "kic": args.kic,
        "system": system,
        "parameters": {name: getattr(args, name) for name in parameter_names},
        "w19_data_dir": str(data_dir.resolve()),
        "interpolator_dir": str(Path(args.interpolator_dir).expanduser().resolve()),
        "eccentricity_database": str(eccentricity_database.resolve()),
        "eccentricity_kde_cache": args.eccentricity_kde_cache,
        "phase_lag_to_qprime": PHASE_LAG_TO_QPRIME,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def trial_checkpoint_path(
    checkpoint_dir: Path, lgq: float, initial_eccentricity: float
) -> Path:
    """Return a stable filename for one point in the q/e grid."""

    return checkpoint_dir / (
        f"lgq_{lgq:.12g}_initial_e_{initial_eccentricity:.12g}.npz"
    )


def write_trial_checkpoint(
    path: Path, trial: dict[str, Any], signature: str
) -> None:
    """Atomically save one complete trial, including its trajectory."""

    trajectory = trial.get("trajectory", {})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {
        "signature": np.asarray(signature),
        "lgq": np.asarray(trial["lgq"], dtype=float),
        "initial_eccentricity": np.asarray(
            trial["initial_eccentricity"], dtype=float
        ),
        "success": np.asarray(trial["success"], dtype=bool),
        "status": np.asarray(trial.get("status", "")),
        "initial_period_day": np.asarray(
            trial.get("initial_period_day", float("nan")), dtype=float
        ),
        "final_period_day": np.asarray(
            trial.get("final_period_day", float("nan")), dtype=float
        ),
        "final_eccentricity": np.asarray(
            trial.get("final_eccentricity", float("nan")), dtype=float
        ),
        "eccentricity_log_likelihood": np.asarray(
            trial.get("eccentricity_log_likelihood", float("-inf")), dtype=float
        ),
        "num_period_evolutions": np.asarray(
            trial.get("num_period_evolutions", 0), dtype=int
        ),
        "num_solver_steps": np.asarray(
            trial.get("num_solver_steps", 0), dtype=int
        ),
        "trajectory_age_gyr": np.asarray(
            trajectory.get("age_gyr", []), dtype=float
        ),
        "trajectory_orbital_period_day": np.asarray(
            trajectory.get("orbital_period_day", []), dtype=float
        ),
        "trajectory_eccentricity": np.asarray(
            trajectory.get("eccentricity", []), dtype=float
        ),
        "solver_final_age_gyr": np.asarray(
            trajectory.get("solver_final_age_gyr", float("nan")), dtype=float
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as output:
            np.savez_compressed(output, **payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_trial_checkpoint(
    path: Path, *, lgq: float, initial_eccentricity: float, signature: str
) -> dict[str, Any]:
    """Load and validate one saved trial."""

    with np.load(path, allow_pickle=False) as archive:
        saved_signature = str(archive["signature"].item())
        if saved_signature != signature:
            raise RuntimeError(
                f"Checkpoint inputs do not match this run: {path}. "
                "Use a different --checkpoint-dir or remove the stale checkpoint."
            )
        saved_lgq = float(archive["lgq"].item())
        saved_eccentricity = float(archive["initial_eccentricity"].item())
        if not (
            math.isclose(saved_lgq, lgq)
            and math.isclose(saved_eccentricity, initial_eccentricity)
        ):
            raise RuntimeError(f"Checkpoint coordinates do not match its name: {path}")
        success = bool(archive["success"].item())
        trial: dict[str, Any] = {
            "lgq": saved_lgq,
            "initial_eccentricity": saved_eccentricity,
            "success": success,
            "status": str(archive["status"].item()),
            "initial_period_day": float(archive["initial_period_day"].item()),
            "final_period_day": float(archive["final_period_day"].item()),
            "final_eccentricity": float(archive["final_eccentricity"].item()),
            "eccentricity_log_likelihood": float(
                archive["eccentricity_log_likelihood"].item()
            ),
            "num_period_evolutions": int(
                archive["num_period_evolutions"].item()
            ),
            "num_solver_steps": int(archive["num_solver_steps"].item()),
        }
        if success:
            age = np.asarray(archive["trajectory_age_gyr"], dtype=float)
            period = np.asarray(
                archive["trajectory_orbital_period_day"], dtype=float
            )
            eccentricity = np.asarray(
                archive["trajectory_eccentricity"], dtype=float
            )
            if age.size < 2 or not (age.size == period.size == eccentricity.size):
                raise RuntimeError(f"Successful checkpoint lacks a trajectory: {path}")
            trial["trajectory"] = {
                "reached_age": True,
                "solver_final_age_gyr": float(
                    archive["solver_final_age_gyr"].item()
                ),
                "final_period_day": trial["final_period_day"],
                "final_eccentricity": trial["final_eccentricity"],
                "num_solver_steps": trial["num_solver_steps"],
                "age_gyr": age,
                "orbital_period_day": period,
                "eccentricity": eccentricity,
            }
    return trial


def recovered_initial_periods(log_paths: list[Path]) -> dict[tuple[float, float], float]:
    """Extract completed q/e root solutions from earlier unbuffered logs."""

    recovered: dict[tuple[float, float], float] = {}
    for log_path in log_paths:
        current_point: tuple[float, float] | None = None
        with log_path.open("r", encoding="utf-8", errors="replace") as source:
            for line in source:
                grid_match = GRID_LOG_PATTERN.match(line.rstrip())
                if grid_match:
                    current_point = (
                        float(grid_match.group("lgq")),
                        float(grid_match.group("ecc")),
                    )
                    continue
                solved_match = SOLVED_LOG_PATTERN.match(line)
                if solved_match and current_point is not None:
                    recovered[current_point] = float(solved_match.group("period"))
                    current_point = None
    return recovered


def constant_q_dissipation(lgq: float) -> dict[str, Any]:
    """Return POET's constant modified-Q' prescription for one stellar zone."""

    return {
        "tidal_frequency_breaks": np.empty(0, dtype=float),
        "spin_frequency_breaks": np.empty(0, dtype=float),
        "tidal_frequency_powers": np.array([0.0]),
        "spin_frequency_powers": np.array([0.0]),
        "age_breaks": np.empty(0, dtype=float),
        "reference_phase_lags": np.array(
            [PHASE_LAG_TO_QPRIME / (10.0**lgq)], dtype=float
        ),
        "inertial_mode_enhancement": 1.0,
        "inertial_mode_sharpness": 10.0,
    }


def make_star(
    mass: float,
    feh: float,
    interpolator: Any,
    dissipation: dict[str, Any],
    args: argparse.Namespace,
) -> EvolvingStar:
    """Create a two-zone POET star dissipating in its convective envelope."""

    star = EvolvingStar(
        mass=mass,
        metallicity=feh,
        wind_strength=args.wind_strength,
        wind_saturation_frequency=args.wind_saturation,
        diff_rot_coupling_timescale=args.core_envelope_coupling_timescale,
        interpolator=interpolator,
    )
    star.select_interpolation_region(star.core_formation_age())
    star.set_dissipation(zone_index=0, **dissipation)
    return star


def disk_locked_secondary_angmom(
    system: dict[str, float],
    interpolator: Any,
    args: argparse.Namespace,
) -> np.ndarray:
    """Find secondary envelope/core angular momenta at disk dispersal."""

    reference_q = float(np.median(args.lgq))
    star = make_star(
        system["secondary_mass"],
        system["feh"],
        interpolator,
        constant_q_dissipation(reference_q),
        args,
    )
    dummy = LockedPlanet(mass=1.0e-6, radius=0.1)
    binary: Binary | None = None
    try:
        start_age = star.core_formation_age()
        if start_age >= args.disk_dissipation_age:
            raise ValueError(
                "Disk-dissipation age must exceed secondary core-formation "
                f"age ({args.disk_dissipation_age} <= {start_age} Gyr)"
            )
        dummy_semimajor = calc_semimajor(
            system["secondary_mass"],
            dummy.mass,
            system["orbital_period_day"],
        )
        dummy.configure(
            age=args.disk_dissipation_age,
            companion_mass=system["secondary_mass"],
            semimajor=dummy_semimajor,
            eccentricity=0.0,
            spin_angmom=np.array([0.0]),
            inclination=None,
            periapsis=None,
            locked_surface=False,
            zero_outer_inclination=True,
            zero_outer_periapsis=True,
        )
        binary = Binary(
            primary=star,
            secondary=dummy,
            initial_orbital_period=system["orbital_period_day"],
            initial_eccentricity=0.0,
            initial_inclination=0.0,
            disk_lock_frequency=2.0 * math.pi / args.disk_lock_period,
            disk_dissipation_age=args.disk_dissipation_age,
            secondary_formation_age=args.disk_dissipation_age,
        )
        binary.configure(
            age=start_age,
            semimajor=float("nan"),
            eccentricity=float("nan"),
            spin_angmom=np.array([0.0]),
            inclination=None,
            periapsis=None,
            evolution_mode="LOCKED_SURFACE_SPIN",
        )
        star.detect_stellar_wind_saturation()
        binary.evolve(
            args.disk_dissipation_age,
            min(
                args.max_time_step,
                (args.disk_dissipation_age - start_age) / 10.0,
            ),
            args.precision,
            np.array([start_age, args.disk_dissipation_age]),
            timeout=args.timeout,
        )
        final_state = binary.final_state()
        if not math.isclose(
            final_state.age,
            args.disk_dissipation_age,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError("Secondary disk-lock evolution ended prematurely")
        return np.array(
            [final_state.envelope_angmom, final_state.core_angmom], dtype=float
        )
    finally:
        if binary is not None:
            binary.delete()
        dummy.delete()
        star.delete()


def output_ages(start_age: float, final_age: float, args: argparse.Namespace) -> np.ndarray:
    """Return logarithmically spaced ages for the saved trajectory."""

    post_disk = np.geomspace(
        args.disk_dissipation_age,
        final_age,
        args.output_samples,
    )
    return np.unique(
        np.concatenate(([start_age, args.disk_dissipation_age], post_disk))
    )


def evolve_binary(
    *,
    initial_period: float,
    initial_eccentricity: float,
    lgq: float,
    system: dict[str, float],
    interpolator: Any,
    secondary_initial_angmom: np.ndarray,
    args: argparse.Namespace,
    save_trajectory: bool,
) -> dict[str, Any]:
    """Evolve one trial and return endpoint information and optional arrays."""

    dissipation = constant_q_dissipation(lgq)
    primary = make_star(
        system["primary_mass"], system["feh"], interpolator, dissipation, args
    )
    secondary = make_star(
        system["secondary_mass"], system["feh"], interpolator, dissipation, args
    )
    binary: Binary | None = None
    try:
        track_limit = min(primary.lifetime(), secondary.lifetime())
        if system["age_gyr"] >= track_limit:
            raise ValueError(
                f"Median age {system['age_gyr']:.6g} Gyr exceeds the shorter "
                f"stellar track ({track_limit:.6g} Gyr)"
            )

        initial_semimajor = calc_semimajor(
            system["primary_mass"],
            system["secondary_mass"],
            initial_period,
        )
        secondary.select_interpolation_region(args.disk_dissipation_age)
        secondary.configure(
            age=args.disk_dissipation_age,
            companion_mass=system["primary_mass"],
            semimajor=initial_semimajor,
            eccentricity=initial_eccentricity,
            spin_angmom=secondary_initial_angmom,
            inclination=np.array([0.0]),
            periapsis=np.array([0.0]),
            locked_surface=False,
            zero_outer_inclination=True,
            zero_outer_periapsis=True,
        )
        secondary.detect_stellar_wind_saturation()

        binary = Binary(
            primary=primary,
            secondary=secondary,
            initial_orbital_period=initial_period,
            initial_eccentricity=initial_eccentricity,
            initial_inclination=0.0,
            disk_lock_frequency=2.0 * math.pi / args.disk_lock_period,
            disk_dissipation_age=args.disk_dissipation_age,
            secondary_formation_age=args.disk_dissipation_age,
        )
        start_age = primary.core_formation_age()
        binary.configure(
            age=start_age,
            semimajor=float("nan"),
            eccentricity=float("nan"),
            spin_angmom=np.array([0.0]),
            inclination=None,
            periapsis=None,
            evolution_mode="LOCKED_SURFACE_SPIN",
        )
        primary.detect_stellar_wind_saturation()
        required_ages = (
            output_ages(start_age, system["age_gyr"], args)
            if save_trajectory
            else np.array([start_age, args.disk_dissipation_age, system["age_gyr"]])
        )
        binary.evolve(
            system["age_gyr"],
            args.max_time_step,
            args.precision,
            required_ages,
            timeout=args.timeout,
        )
        final_state = binary.final_state()
        reached_age = math.isclose(
            final_state.age,
            system["age_gyr"],
            rel_tol=0.0,
            abs_tol=max(1.0e-8, args.max_time_step * 1.0e-5),
        )
        final_period = (
            float(binary.orbital_period(final_state.semimajor))
            if math.isfinite(final_state.semimajor)
            else float("nan")
        )
        result: dict[str, Any] = {
            "reached_age": reached_age,
            "solver_final_age_gyr": float(final_state.age),
            "final_period_day": final_period,
            "final_eccentricity": float(final_state.eccentricity),
            "num_solver_steps": int(binary.num_evolution_steps),
        }
        if save_trajectory:
            evolution = binary.get_evolution(required_ages_only=True)
            finite = (
                np.isfinite(evolution.age)
                & np.isfinite(evolution.semimajor)
                & np.isfinite(evolution.eccentricity)
            )
            result.update(
                age_gyr=np.asarray(evolution.age[finite], dtype=float),
                orbital_period_day=np.asarray(
                    binary.orbital_period(evolution.semimajor[finite]), dtype=float
                ),
                eccentricity=np.asarray(evolution.eccentricity[finite], dtype=float),
            )
        return result
    finally:
        if binary is not None:
            binary.delete()
        primary.delete()
        secondary.delete()


def solve_initial_period(
    *,
    lgq: float,
    initial_eccentricity: float,
    system: dict[str, float],
    interpolator: Any,
    secondary_initial_angmom: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any], int]:
    """Solve for the initial period that reaches the observed final period."""

    target_period = system["orbital_period_day"]
    evaluations = 0
    cache: dict[float, tuple[float, dict[str, Any] | None]] = {}

    def objective(initial_period: float) -> float:
        nonlocal evaluations
        key = round(float(initial_period), 12)
        if key in cache:
            return cache[key][0]
        evaluations += 1
        try:
            endpoint = evolve_binary(
                initial_period=initial_period,
                initial_eccentricity=initial_eccentricity,
                lgq=lgq,
                system=system,
                interpolator=interpolator,
                secondary_initial_angmom=secondary_initial_angmom,
                args=args,
                save_trajectory=False,
            )
            if endpoint["reached_age"] and math.isfinite(endpoint["final_period_day"]):
                value = endpoint["final_period_day"] - target_period
            else:
                value = -target_period
        except Exception:
            endpoint = None
            value = -target_period
        cache[key] = (float(value), endpoint)
        return float(value)

    def report_point(period: float, value: float) -> None:
        print(
            f"    bracket P_i={period:.8g} d -> delta P_f={value:+.5g} d",
            flush=True,
        )

    period = target_period
    value = objective(period)
    report_point(period, value)
    root = float("nan")
    if abs(value) <= args.period_tolerance:
        root = period
    else:
        # Orbital evolution often has d(P_final)/d(P_initial) close to one.
        # Use that to make a cheap first estimate on the correct side of the
        # target.  This handles both orbital decay (search upward) and orbital
        # expansion (search downward), unlike the old one-sided bracket.
        next_period = min(
            max(period - value, args.min_initial_period),
            args.max_initial_period,
        )
        if math.isclose(next_period, period, rel_tol=0.0, abs_tol=1.0e-12):
            next_period = (
                period * args.period_search_factor
                if value < 0.0
                else period / args.period_search_factor
            )

        previous_period, previous_value = period, value
        for _ in range(args.max_bracket_steps):
            period = next_period
            if not args.min_initial_period <= period <= args.max_initial_period:
                break
            value = objective(period)
            report_point(period, value)
            if abs(value) <= args.period_tolerance:
                root = period
                break
            if value * previous_value < 0.0:
                lower_period, upper_period = sorted((previous_period, period))
                root = brentq(
                    objective,
                    lower_period,
                    upper_period,
                    xtol=args.period_tolerance,
                    rtol=1.0e-12,
                    maxiter=args.max_root_iterations,
                )
                break

            previous_period, previous_value = period, value
            next_period = (
                period * args.period_search_factor
                if value < 0.0
                else period / args.period_search_factor
            )
            next_period = min(
                max(next_period, args.min_initial_period),
                args.max_initial_period,
            )
            if math.isclose(next_period, period, rel_tol=0.0, abs_tol=1.0e-12):
                break

    if not math.isfinite(root):
        raise RuntimeError(
            "Could not bracket the initial period: "
            f"last point=({period:.8g} d, delta P_f={value:+.8g} d), "
            f"allowed range=[{args.min_initial_period}, "
            f"{args.max_initial_period}] d"
        )

    trajectory = evolve_binary(
        initial_period=float(root),
        initial_eccentricity=initial_eccentricity,
        lgq=lgq,
        system=system,
        interpolator=interpolator,
        secondary_initial_angmom=secondary_initial_angmom,
        args=args,
        save_trajectory=True,
    )
    if not trajectory["reached_age"]:
        raise RuntimeError(
            f"Best-fit trajectory stopped at {trajectory['solver_final_age_gyr']} Gyr"
        )
    return float(root), trajectory, evaluations + 1


def integration_cell_widths(values: np.ndarray) -> np.ndarray:
    """Return midpoint-rule cell widths for a sorted one-dimensional grid."""

    if values.size == 1:
        return np.ones(1)
    widths = np.empty_like(values)
    widths[1:-1] = 0.5 * (values[2:] - values[:-2])
    widths[0] = 0.5 * (values[1] - values[0])
    widths[-1] = 0.5 * (values[-1] - values[-2])
    return widths


def weighted_quantile(
    values: np.ndarray, probabilities: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """Evaluate quantiles of a normalized discrete distribution."""

    order = np.argsort(values)
    ordered_values = values[order]
    ordered_probabilities = probabilities[order]
    total_probability = np.sum(ordered_probabilities)
    # Treat each quadrature weight as representing a cell centered on its lgQ
    # value.  Using the right edge of every cell biases all quantiles downward,
    # including the median of a symmetric distribution.
    cumulative_centers = (
        np.cumsum(ordered_probabilities) - 0.5 * ordered_probabilities
    ) / total_probability
    return np.interp(
        quantiles,
        cumulative_centers,
        ordered_values,
        left=ordered_values[0],
        right=ordered_values[-1],
    )


def grid_lgq_distribution(
    lgq_values: np.ndarray,
    initial_eccentricities: np.ndarray,
    trials: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Marginalize endpoint likelihood over the uniform e_i prior on [0, 0.8]."""

    log_likelihood = np.full(lgq_values.size, -np.inf)
    if initial_eccentricities.size == 1:
        eccentricity_log_weights = np.zeros(1)
    else:
        eccentricity_widths = integration_cell_widths(initial_eccentricities)
        eccentricity_log_weights = np.log(
            eccentricity_widths / np.sum(eccentricity_widths)
        )
    for index, lgq in enumerate(lgq_values):
        integrand = np.full(initial_eccentricities.size, -np.inf)
        for eccentricity_index, initial_eccentricity in enumerate(
            initial_eccentricities
        ):
            matching = [
                trial
                for trial in trials
                if (
                    trial["success"]
                    and math.isclose(trial["lgq"], lgq)
                    and math.isclose(
                        trial["initial_eccentricity"], initial_eccentricity
                    )
                )
            ]
            if matching:
                integrand[eccentricity_index] = matching[
                    0
                ]["eccentricity_log_likelihood"]
        finite_integrand = np.isfinite(integrand)
        if np.any(finite_integrand):
            log_likelihood[index] = logsumexp(
                integrand[finite_integrand]
                + eccentricity_log_weights[finite_integrand]
            )

    finite = np.isfinite(log_likelihood)
    if not np.any(finite):
        raise RuntimeError("No successful trial has a finite eccentricity likelihood")
    cell_width = integration_cell_widths(lgq_values)
    log_weights = np.full_like(log_likelihood, -np.inf)
    log_weights[finite] = log_likelihood[finite] + np.log(cell_width[finite])
    probabilities = np.exp(log_weights - logsumexp(log_weights[finite]))
    return log_likelihood, probabilities


def summarize_lgq_distribution(
    *,
    lgq_values: np.ndarray,
    log_likelihood: np.ndarray,
    probabilities: np.ndarray,
    posterior_samples: np.ndarray | None,
    posterior_source: str,
    confidence: float,
    initial_eccentricities: np.ndarray,
) -> dict[str, Any]:
    """Calculate bounds and state whether the grid actually supports them."""

    tail = 0.5 * (1.0 - confidence)
    grid_covers_e_prior = (
        initial_eccentricities.size >= 3
        and math.isclose(float(initial_eccentricities[0]), 0.0)
        and math.isclose(float(initial_eccentricities[-1]), 0.8)
    )
    if posterior_samples is not None:
        lower, median, upper = np.quantile(
            posterior_samples, [tail, 0.5, 1.0 - tail]
        )
        return {
            "available": True,
            "source": posterior_source,
            "confidence": confidence,
            "lower_lgq": float(lower),
            "median_lgq": float(median),
            "upper_lgq": float(upper),
            "lower_grid_edge_limited": False,
            "upper_grid_edge_limited": False,
            "initial_eccentricity_prior_integrated": True,
            "note": "Bounds are empirical quantiles of lgQ_min from the MCMC blobs.",
        }

    if lgq_values.size < 3 or not grid_covers_e_prior:
        return {
            "available": False,
            "source": posterior_source,
            "confidence": confidence,
            "lower_lgq": None,
            "median_lgq": None,
            "upper_lgq": None,
            "lower_grid_edge_limited": None,
            "upper_grid_edge_limited": None,
            "initial_eccentricity_prior_integrated": grid_covers_e_prior,
            "note": (
                "Bounds require at least three lgQ values and an initial-"
                "eccentricity grid spanning the uniform prior from 0 to 0.8."
            ),
        }

    lower, median, upper = weighted_quantile(
        lgq_values,
        probabilities,
        np.array([tail, 0.5, 1.0 - tail]),
    )
    finite = np.isfinite(log_likelihood)
    maximum_log_likelihood = float(np.max(log_likelihood[finite]))
    edge_relative_likelihood = np.exp(
        np.array([log_likelihood[0], log_likelihood[-1]])
        - maximum_log_likelihood
    )
    edge_limited = edge_relative_likelihood > 0.05
    return {
        "available": True,
        "source": posterior_source,
        "confidence": confidence,
        "lower_lgq": float(lower),
        "median_lgq": float(median),
        "upper_lgq": float(upper),
        "lower_grid_edge_limited": bool(edge_limited[0]),
        "upper_grid_edge_limited": bool(edge_limited[1]),
        "initial_eccentricity_prior_integrated": True,
        "edge_likelihood_threshold": 0.05,
        "lower_edge_relative_likelihood": float(edge_relative_likelihood[0]),
        "upper_edge_relative_likelihood": float(edge_relative_likelihood[1]),
        "note": (
            "Grid-posterior quantiles use a uniform lgQ prior and integrate "
            "the uniform initial-eccentricity prior on [0, 0.8]. An edge-"
            "limited flag means the lgQ grid must be extended before treating "
            "that side as a data-driven limit."
        ),
    }


def load_mcmc_lgq(
    path: Path | None, burn_in: int, thin: int
) -> tuple[np.ndarray | None, str | None]:
    """Load lgQ_min from emcee blobs; return None for absent/empty chains."""

    if path is None or not path.is_file():
        return None, None
    collected: list[np.ndarray] = []
    try:
        with h5py.File(path, "r") as archive:
            for group_name in archive:
                group = archive[group_name]
                if not isinstance(group, h5py.Group) or "blobs" not in group:
                    continue
                blobs = group["blobs"]
                if blobs.shape[0] <= burn_in or blobs.dtype.names is None:
                    continue
                if "lgQ_min" not in blobs.dtype.names:
                    continue
                values = np.asarray(blobs[burn_in::thin]["lgQ_min"], dtype=float)
                collected.append(values.ravel())
    except (OSError, ValueError):
        return None, None
    if not collected:
        return None, None
    values = np.concatenate(collected)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    return values, str(path.resolve())


def save_results(
    *,
    output_dir: Path,
    kic: int,
    trials: list[dict[str, Any]],
    lgq_values: np.ndarray,
    log_likelihood: np.ndarray,
    probabilities: np.ndarray,
    bounds: dict[str, Any],
    config: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    """Write trial summaries, every retained trajectory, and configuration."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"KIC{kic}_constant_q_scan"
    summary_path = output_dir / f"{stem}_summary.csv"
    track_path = output_dir / f"{stem}_trajectories.csv"
    bounds_path = output_dir / f"{stem}_lgq_bounds.json"
    config_path = output_dir / f"{stem}_config.json"

    q_index = {float(value): index for index, value in enumerate(lgq_values)}
    fields = [
        "lgq",
        "initial_eccentricity",
        "success",
        "status",
        "initial_period_day",
        "final_period_day",
        "final_eccentricity",
        "eccentricity_log_likelihood",
        "lgq_log_likelihood",
        "lgq_probability",
        "num_period_evolutions",
        "num_solver_steps",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for trial in trials:
            row = {name: trial.get(name, "") for name in fields}
            index = q_index[float(trial["lgq"])]
            row["lgq_log_likelihood"] = log_likelihood[index]
            row["lgq_probability"] = probabilities[index]
            writer.writerow(row)

    with track_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "lgq",
                "initial_eccentricity",
                "age_gyr",
                "orbital_period_day",
                "eccentricity",
            ]
        )
        for trial in trials:
            if not trial["success"]:
                continue
            trajectory = trial["trajectory"]
            for age, period, eccentricity in zip(
                trajectory["age_gyr"],
                trajectory["orbital_period_day"],
                trajectory["eccentricity"],
            ):
                writer.writerow(
                    [
                        trial["lgq"],
                        trial["initial_eccentricity"],
                        age,
                        period,
                        eccentricity,
                    ]
                )

    with config_path.open("w", encoding="utf-8") as output:
        json.dump(config, output, indent=2, sort_keys=True)
    with bounds_path.open("w", encoding="utf-8") as output:
        json.dump(bounds, output, indent=2, sort_keys=True)
    return summary_path, track_path, bounds_path, config_path


def plot_results(
    *,
    output_dir: Path,
    kic: int,
    system: dict[str, float],
    observed_eccentricity: np.ndarray,
    trials: list[dict[str, Any]],
    lgq_values: np.ndarray,
    probabilities: np.ndarray,
    posterior_samples: np.ndarray | None,
    posterior_source: str,
    bounds: dict[str, Any],
    confidence: float,
    linear_age: bool,
) -> Path:
    """Plot all successful evolutions and the final lgQ distribution."""

    quantiles = (
        np.array(
            [bounds["lower_lgq"], bounds["median_lgq"], bounds["upper_lgq"]]
        )
        if bounds["available"]
        else None
    )

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.3), constrained_layout=True)
    period_axis, eccentricity_axis, posterior_axis = axes
    q_norm = colors.Normalize(vmin=float(lgq_values.min()), vmax=float(lgq_values.max()))
    color_map = plt.get_cmap("viridis")
    unique_e = sorted({trial["initial_eccentricity"] for trial in trials})
    line_styles = ["-", "--", "-.", ":"]

    for trial in trials:
        if not trial["success"]:
            continue
        trajectory = trial["trajectory"]
        line_style = line_styles[unique_e.index(trial["initial_eccentricity"]) % 4]
        color = color_map(q_norm(trial["lgq"]))
        period_axis.plot(
            trajectory["age_gyr"],
            trajectory["orbital_period_day"],
            color=color,
            linestyle=line_style,
            linewidth=1.25,
            alpha=0.82,
        )
        eccentricity_axis.plot(
            trajectory["age_gyr"],
            trajectory["eccentricity"],
            color=color,
            linestyle=line_style,
            linewidth=1.25,
            alpha=0.82,
        )

    period_axis.axhline(
        system["orbital_period_day"],
        color="black",
        linewidth=1.0,
        linestyle=":",
        label="observed median",
    )
    eccentricity_limits = np.quantile(observed_eccentricity, [0.158655, 0.841345])
    eccentricity_axis.axhspan(
        eccentricity_limits[0],
        eccentricity_limits[1],
        color="black",
        alpha=0.10,
        label="W19 central 68.27%",
    )
    eccentricity_axis.axhline(
        system["eccentricity"], color="black", linewidth=1.0, linestyle=":"
    )

    if posterior_samples is None:
        if lgq_values.size > 1:
            bar_width = 0.75 * float(np.min(np.diff(lgq_values)))
        else:
            bar_width = 0.25
        posterior_axis.bar(
            lgq_values,
            probabilities,
            width=bar_width,
            color=[color_map(q_norm(value)) for value in lgq_values],
            edgecolor="black",
            linewidth=0.4,
        )
        posterior_axis.set_ylabel("normalized grid probability")
    else:
        posterior_axis.hist(
            posterior_samples,
            bins="auto",
            density=True,
            color="0.35",
            alpha=0.75,
        )
        posterior_axis.set_ylabel("MCMC posterior density")

    if quantiles is not None:
        for value, style in zip(quantiles, ["--", "-", "--"]):
            posterior_axis.axvline(
                value, color="firebrick", linestyle=style, linewidth=1.4
            )

    if not linear_age:
        period_axis.set_xscale("log")
        eccentricity_axis.set_xscale("log")
    period_axis.set_xlabel("age (Gyr)")
    period_axis.set_ylabel("orbital period (day)")
    eccentricity_axis.set_xlabel("age (Gyr)")
    eccentricity_axis.set_ylabel("eccentricity")
    posterior_axis.set_xlabel(r"$\lg Q'$ (constant for both stars)")
    period_axis.legend(loc="best", fontsize="small")
    eccentricity_handles, eccentricity_labels = (
        eccentricity_axis.get_legend_handles_labels()
    )
    eccentricity_handles.extend(
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle=line_styles[index % 4],
            label=rf"$e_i={value:g}$",
        )
        for index, value in enumerate(unique_e)
    )
    eccentricity_labels.extend(rf"$e_i={value:g}$" for value in unique_e)
    eccentricity_axis.legend(
        eccentricity_handles,
        eccentricity_labels,
        loc="best",
        fontsize="small",
    )
    period_axis.grid(alpha=0.2)
    eccentricity_axis.grid(alpha=0.2)
    posterior_axis.grid(alpha=0.2)

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=q_norm, cmap=color_map),
        ax=[period_axis, eccentricity_axis],
        location="bottom",
        shrink=0.72,
        pad=0.02,
        aspect=35,
    )
    colorbar.set_label(r"tested $\lg Q'$")

    percent = 100.0 * confidence
    if quantiles is None:
        distribution_title = (
            f"KIC {kic}: SMOKE TEST — lgQ' bounds unavailable; "
            "run a multi-lgQ grid and integrate 0 <= e_i <= 0.8"
        )
    else:
        edge_limited = (
            bounds["lower_grid_edge_limited"]
            or bounds["upper_grid_edge_limited"]
        )
        distribution_title = (
            f"KIC {kic}: {posterior_source} lgQ' distribution — "
            f"median={quantiles[1]:.3f}, central {percent:.2f}% "
            f"=[{quantiles[0]:.3f}, {quantiles[2]:.3f}]"
            + (" (GRID-EDGE LIMITED)" if edge_limited else "")
        )
    fig.suptitle(
        (
            distribution_title
            + "\n"
            f"M1={system['primary_mass']:.3f} Msun, "
            f"M2={system['secondary_mass']:.3f} Msun, "
            f"[Fe/H]={system['feh']:+.3f}, age={system['age_gyr']:.3f} Gyr"
        ),
        fontsize=12,
    )

    figure_path = output_dir / f"KIC{kic}_constant_q_scan.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path


_MASS_GRID_RUNTIME: dict[str, Any] = {}
_MASS_GRID_SPIN_CACHE: dict[float, np.ndarray] = {}


def _grid_from_spec(
    spec: list[float] | None,
    *,
    default_bounds: tuple[float, float],
    default_count: int,
    name: str,
) -> np.ndarray:
    """Build an inclusive linear grid from MIN, MAX, COUNT."""

    if spec is None:
        lower, upper = default_bounds
        count = default_count
    else:
        lower, upper, raw_count = spec
        count = int(raw_count)
        if not math.isclose(raw_count, count):
            raise ValueError(f"{name} COUNT must be an integer, got {raw_count}")
    if count < 1:
        raise ValueError(f"{name} COUNT must be positive")
    if lower > upper:
        raise ValueError(f"{name} MIN must not exceed MAX")
    if count == 1 and not math.isclose(lower, upper):
        raise ValueError(f"{name} with COUNT=1 requires MIN=MAX")
    return np.linspace(lower, upper, count, dtype=float)


def mass_grid_checkpoint_signature(
    *,
    args: argparse.Namespace,
    base_system: dict[str, float],
    data_dir: Path,
    eccentricity_database: Path,
) -> str:
    """Identify all non-grid inputs affecting a mass/eccentricity trial."""

    parameter_names = (
        "mass_grid_lgq",
        "mass_grid_period_mode",
        "mass_grid_initial_period",
        "disk_dissipation_age",
        "disk_lock_period",
        "wind_strength",
        "wind_saturation",
        "core_envelope_coupling_timescale",
        "max_time_step",
        "precision",
        "eccentricity_precision",
        "timeout",
        "output_samples",
        "period_tolerance",
        "period_search_factor",
        "min_initial_period",
        "max_initial_period",
        "max_bracket_steps",
        "max_root_iterations",
    )
    fixed_system = {
        key: value
        for key, value in base_system.items()
        if key not in ("primary_mass", "secondary_mass")
    }
    payload = {
        "checkpoint_format": 2,
        "kic": args.kic,
        "fixed_system": fixed_system,
        "parameters": {name: getattr(args, name) for name in parameter_names},
        "w19_data_dir": str(data_dir.resolve()),
        "interpolator_dir": str(Path(args.interpolator_dir).expanduser().resolve()),
        "eccentricity_database": str(eccentricity_database.resolve()),
        "phase_lag_to_qprime": PHASE_LAG_TO_QPRIME,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def mass_grid_checkpoint_path(
    checkpoint_dir: Path,
    primary_mass: float,
    secondary_mass: float,
    initial_eccentricity: float,
) -> Path:
    """Return a stable, coordinate-based checkpoint filename."""

    return checkpoint_dir / (
        f"m1_{primary_mass:.12g}_m2_{secondary_mass:.12g}_"
        f"initial_e_{initial_eccentricity:.12g}.npz"
    )


def write_mass_grid_checkpoint(
    path: Path, trial: dict[str, Any], signature: str
) -> None:
    """Atomically save one mass/eccentricity trial and its trajectory."""

    trajectory = trial.get("trajectory", {})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {
        "signature": np.asarray(signature),
        "primary_index": np.asarray(trial["primary_index"], dtype=int),
        "secondary_index": np.asarray(trial["secondary_index"], dtype=int),
        "eccentricity_index": np.asarray(trial["eccentricity_index"], dtype=int),
        "primary_mass": np.asarray(trial["primary_mass"], dtype=float),
        "secondary_mass": np.asarray(trial["secondary_mass"], dtype=float),
        "initial_eccentricity": np.asarray(
            trial["initial_eccentricity"], dtype=float
        ),
        "success": np.asarray(trial["success"], dtype=bool),
        "status": np.asarray(trial.get("status", "")),
        "initial_period_day": np.asarray(
            trial.get("initial_period_day", float("nan")), dtype=float
        ),
        "final_period_day": np.asarray(
            trial.get("final_period_day", float("nan")), dtype=float
        ),
        "final_eccentricity": np.asarray(
            trial.get("final_eccentricity", float("nan")), dtype=float
        ),
        "period_error_day": np.asarray(
            trial.get("period_error_day", float("nan")), dtype=float
        ),
        "num_period_evolutions": np.asarray(
            trial.get("num_period_evolutions", 0), dtype=int
        ),
        "num_solver_steps": np.asarray(
            trial.get("num_solver_steps", 0), dtype=int
        ),
        "elapsed_seconds": np.asarray(
            trial.get("elapsed_seconds", float("nan")), dtype=float
        ),
        "evolution_seconds": np.asarray(
            trial.get("evolution_seconds", float("nan")), dtype=float
        ),
        "evolution_started_at": np.asarray(
            trial.get("evolution_started_at", float("nan")), dtype=float
        ),
        "evolution_finished_at": np.asarray(
            trial.get("evolution_finished_at", float("nan")), dtype=float
        ),
        "trajectory_age_gyr": np.asarray(
            trajectory.get("age_gyr", []), dtype=float
        ),
        "trajectory_orbital_period_day": np.asarray(
            trajectory.get("orbital_period_day", []), dtype=float
        ),
        "trajectory_eccentricity": np.asarray(
            trajectory.get("eccentricity", []), dtype=float
        ),
        "solver_final_age_gyr": np.asarray(
            trajectory.get("solver_final_age_gyr", float("nan")), dtype=float
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as output:
            np.savez_compressed(output, **payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_mass_grid_checkpoint(
    path: Path,
    *,
    task: tuple[int, int, int, float, float, float],
    signature: str,
) -> dict[str, Any]:
    """Load and validate a mass/eccentricity checkpoint."""

    i1, i2, ie, primary_mass, secondary_mass, initial_eccentricity = task
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["signature"].item()) != signature:
            raise RuntimeError(
                f"Checkpoint inputs do not match this run: {path}. Use a "
                "different --checkpoint-dir or move the stale checkpoint."
            )
        coordinates = (
            float(archive["primary_mass"].item()),
            float(archive["secondary_mass"].item()),
            float(archive["initial_eccentricity"].item()),
        )
        expected = (primary_mass, secondary_mass, initial_eccentricity)
        if not all(
            math.isclose(left, right)
            for left, right in zip(coordinates, expected)
        ):
            raise RuntimeError(
                f"Checkpoint coordinates do not match its name: {path}"
            )
        trial: dict[str, Any] = {
            "primary_index": i1,
            "secondary_index": i2,
            "eccentricity_index": ie,
            "primary_mass": coordinates[0],
            "secondary_mass": coordinates[1],
            "initial_eccentricity": coordinates[2],
            "success": bool(archive["success"].item()),
            "status": str(archive["status"].item()),
            "initial_period_day": float(archive["initial_period_day"].item()),
            "final_period_day": float(archive["final_period_day"].item()),
            "final_eccentricity": float(archive["final_eccentricity"].item()),
            "period_error_day": float(archive["period_error_day"].item()),
            "num_period_evolutions": int(
                archive["num_period_evolutions"].item()
            ),
            "num_solver_steps": int(archive["num_solver_steps"].item()),
            "elapsed_seconds": float(archive["elapsed_seconds"].item()),
            "evolution_seconds": float(archive["evolution_seconds"].item()),
            "evolution_started_at": float(
                archive["evolution_started_at"].item()
            ),
            "evolution_finished_at": float(
                archive["evolution_finished_at"].item()
            ),
            "resumed": True,
        }
        if trial["success"]:
            trial["trajectory"] = {
                "solver_final_age_gyr": float(
                    archive["solver_final_age_gyr"].item()
                ),
                "age_gyr": np.asarray(archive["trajectory_age_gyr"], dtype=float),
                "orbital_period_day": np.asarray(
                    archive["trajectory_orbital_period_day"], dtype=float
                ),
                "eccentricity": np.asarray(
                    archive["trajectory_eccentricity"], dtype=float
                ),
            }
    return trial


def _run_mass_grid_trial(
    task: tuple[int, int, int, float, float, float]
) -> dict[str, Any]:
    """Worker entry point for one independent POET grid trial."""

    runtime = _MASS_GRID_RUNTIME
    if not runtime:
        raise RuntimeError("Mass-grid worker started without inherited POET state")
    i1, i2, ie, primary_mass, secondary_mass, initial_eccentricity = task
    args = runtime["args"]
    system = dict(runtime["base_system"])
    system.update(primary_mass=primary_mass, secondary_mass=secondary_mass)
    trial: dict[str, Any] = {
        "primary_index": i1,
        "secondary_index": i2,
        "eccentricity_index": ie,
        "primary_mass": primary_mass,
        "secondary_mass": secondary_mass,
        "initial_eccentricity": initial_eccentricity,
        "success": False,
    }
    started = time.monotonic()
    evolution_timer: float | None = None
    try:
        spin_key = round(secondary_mass, 12)
        if spin_key not in _MASS_GRID_SPIN_CACHE:
            _MASS_GRID_SPIN_CACHE[spin_key] = disk_locked_secondary_angmom(
                system, runtime["interpolator"], args
            )
        secondary_initial_angmom = _MASS_GRID_SPIN_CACHE[spin_key]
        # The serial solver prints every root-bracketing evaluation. Suppress
        # that worker-local chatter so parallel progress remains readable.
        trial["evolution_started_at"] = time.time()
        evolution_timer = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            if args.mass_grid_period_mode == "solve":
                initial_period, trajectory, evaluations = solve_initial_period(
                    lgq=args.mass_grid_lgq,
                    initial_eccentricity=initial_eccentricity,
                    system=system,
                    interpolator=runtime["interpolator"],
                    secondary_initial_angmom=secondary_initial_angmom,
                    args=args,
                )
            else:
                initial_period = runtime["fixed_initial_period"]
                trajectory = evolve_binary(
                    initial_period=initial_period,
                    initial_eccentricity=initial_eccentricity,
                    lgq=args.mass_grid_lgq,
                    system=system,
                    interpolator=runtime["interpolator"],
                    secondary_initial_angmom=secondary_initial_angmom,
                    args=args,
                    save_trajectory=True,
                )
                evaluations = 1
                if not trajectory["reached_age"]:
                    raise RuntimeError(
                        "Evolution stopped at "
                        f"{trajectory['solver_final_age_gyr']:.8g} Gyr"
                    )
        trial["evolution_seconds"] = time.perf_counter() - evolution_timer
        trial["evolution_finished_at"] = time.time()
        period_error = (
            trajectory["final_period_day"] - system["orbital_period_day"]
        )
        trial.update(
            success=True,
            status="ok",
            initial_period_day=initial_period,
            final_period_day=trajectory["final_period_day"],
            final_eccentricity=trajectory["final_eccentricity"],
            period_error_day=period_error,
            num_period_evolutions=evaluations,
            num_solver_steps=trajectory["num_solver_steps"],
            trajectory=trajectory,
        )
    except Exception as error:  # Keep all other grid points running.
        if evolution_timer is not None:
            trial["evolution_seconds"] = time.perf_counter() - evolution_timer
            trial["evolution_finished_at"] = time.time()
        trial.update(
            status=f"{type(error).__name__}: {error}",
            initial_period_day=float("nan"),
            final_period_day=float("nan"),
            final_eccentricity=float("nan"),
            period_error_day=float("nan"),
            num_period_evolutions=0,
            num_solver_steps=0,
        )
    trial["elapsed_seconds"] = time.monotonic() - started
    trial.setdefault("evolution_seconds", float("nan"))
    trial.setdefault("evolution_started_at", float("nan"))
    trial.setdefault("evolution_finished_at", float("nan"))
    trial["resumed"] = False
    return trial


def save_mass_grid_results(
    *,
    output_dir: Path,
    kic: int,
    primary_masses: np.ndarray,
    secondary_masses: np.ndarray,
    eccentricities: np.ndarray,
    trials: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Save a tidy CSV and dense endpoint arrays for the 3-D grid."""

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"KIC{kic}_mass_eccentricity_grid"
    summary_path = prefix.with_name(prefix.name + "_summary.csv")
    fields = [
        "primary_index",
        "secondary_index",
        "eccentricity_index",
        "primary_mass",
        "secondary_mass",
        "initial_eccentricity",
        "success",
        "status",
        "initial_period_day",
        "final_period_day",
        "final_eccentricity",
        "period_error_day",
        "num_period_evolutions",
        "num_solver_steps",
        "evolution_seconds",
        "elapsed_seconds",
    ]
    ordered = sorted(
        trials,
        key=lambda trial: (
            trial["primary_index"],
            trial["secondary_index"],
            trial["eccentricity_index"],
        ),
    )
    with summary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for trial in ordered:
            writer.writerow({field: trial.get(field) for field in fields})

    shape = (primary_masses.size, secondary_masses.size, eccentricities.size)
    success = np.zeros(shape, dtype=bool)
    initial_period = np.full(shape, np.nan)
    final_period = np.full(shape, np.nan)
    final_eccentricity = np.full(shape, np.nan)
    period_error = np.full(shape, np.nan)
    evolution_calls = np.zeros(shape, dtype=int)
    solver_steps = np.zeros(shape, dtype=int)
    evolution_seconds = np.full(shape, np.nan)
    elapsed_seconds = np.full(shape, np.nan)
    status = np.full(shape, "not_run", dtype="U512")
    for trial in ordered:
        index = (
            trial["primary_index"],
            trial["secondary_index"],
            trial["eccentricity_index"],
        )
        success[index] = trial["success"]
        initial_period[index] = trial["initial_period_day"]
        final_period[index] = trial["final_period_day"]
        final_eccentricity[index] = trial["final_eccentricity"]
        period_error[index] = trial["period_error_day"]
        evolution_calls[index] = trial["num_period_evolutions"]
        solver_steps[index] = trial["num_solver_steps"]
        evolution_seconds[index] = trial["evolution_seconds"]
        elapsed_seconds[index] = trial["elapsed_seconds"]
        status[index] = trial["status"]

    arrays_path = prefix.with_name(prefix.name + "_endpoints.npz")
    temporary = arrays_path.with_name(f".{arrays_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output,
            primary_masses=primary_masses,
            secondary_masses=secondary_masses,
            initial_eccentricities=eccentricities,
            success=success,
            status=status,
            initial_period_day=initial_period,
            final_period_day=final_period,
            final_eccentricity=final_eccentricity,
            period_error_day=period_error,
            num_period_evolutions=evolution_calls,
            num_solver_steps=solver_steps,
            evolution_seconds=evolution_seconds,
            elapsed_seconds=elapsed_seconds,
        )
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, arrays_path)

    config_path = prefix.with_name(prefix.name + "_config.json")
    with config_path.open("w", encoding="utf-8") as output:
        json.dump(config, output, indent=2, sort_keys=True)
        output.write("\n")
    return summary_path, arrays_path, config_path


def run_mass_eccentricity_grid(
    *,
    args: argparse.Namespace,
    samples: dict[str, np.ndarray],
    base_system: dict[str, float],
    data_dir: Path,
    eccentricity_database: Path,
) -> None:
    """Run a 3-D (M1, M2, e_initial) grid with process parallelism."""

    mass_bounds = tuple(
        float(value)
        for value in np.quantile(samples["primary_mass"], args.mass_quantiles)
    )
    secondary_bounds = tuple(
        float(value)
        for value in np.quantile(samples["secondary_mass"], args.mass_quantiles)
    )
    primary_masses = _grid_from_spec(
        args.primary_mass_grid,
        default_bounds=mass_bounds,
        default_count=args.mass_grid_size,
        name="--primary-mass-grid",
    )
    secondary_masses = _grid_from_spec(
        args.secondary_mass_grid,
        default_bounds=secondary_bounds,
        default_count=args.mass_grid_size,
        name="--secondary-mass-grid",
    )
    initial_eccentricities = _grid_from_spec(
        args.mass_grid_eccentricity,
        default_bounds=(0.0, 0.8),
        default_count=args.mass_grid_size,
        name="--mass-grid-eccentricity",
    )
    if np.any((initial_eccentricities < 0.0) | (initial_eccentricities > 0.8)):
        raise ValueError("The initial-eccentricity grid must lie in [0, 0.8]")

    tasks = [
        (i1, i2, ie, float(m1), float(m2), float(eccentricity))
        for i1, m1 in enumerate(primary_masses)
        for i2, m2 in enumerate(secondary_masses)
        for ie, eccentricity in enumerate(initial_eccentricities)
    ]
    total = len(tasks)
    fixed_initial_period = (
        base_system["orbital_period_day"]
        if args.mass_grid_initial_period is None
        else args.mass_grid_initial_period
    )
    print(
        f"Mass/eccentricity grid: {primary_masses.size} x "
        f"{secondary_masses.size} x {initial_eccentricities.size} = {total} trials",
        flush=True,
    )
    print(
        f"M1=[{primary_masses[0]:.8g}, {primary_masses[-1]:.8g}], "
        f"M2=[{secondary_masses[0]:.8g}, {secondary_masses[-1]:.8g}] Msun, "
        f"e_i=[{initial_eccentricities[0]:.8g}, "
        f"{initial_eccentricities[-1]:.8g}], lgQ={args.mass_grid_lgq:.8g}",
        flush=True,
    )
    if args.mass_grid_period_mode == "fixed":
        print(
            f"Fixed P_i={fixed_initial_period:.8g} d: exactly one POET evolution "
            "per grid point.",
            flush=True,
        )
    else:
        print(
            "Initial period will be solved per point: each grid trial requires "
            "multiple POET evolutions.",
            flush=True,
        )
    if args.mass_grid_dry_run:
        print("Dry run requested; POET was not initialized.", flush=True)
        return

    print("Preparing the shared POET eccentricity expansion...", flush=True)
    evolution_library.prepare_eccentricity_expansion(
        str(eccentricity_database).encode("ascii"),
        args.eccentricity_precision,
        args.preload_eccentricity,
        True,
    )
    manager = StellarEvolutionManager(str(Path(args.interpolator_dir).expanduser()))
    interpolator = manager.get_interpolator_by_name("default")
    MESAInterpolator.set_quantity_lower_limit("iconv", 1.0e-3)
    mass_min, mass_max = interpolator.mass_range()
    feh_min, feh_max = interpolator.feh_range()
    if primary_masses[0] < mass_min or primary_masses[-1] > mass_max:
        raise ValueError(
            f"Primary-mass grid lies outside POET's [{mass_min}, {mass_max}] Msun range"
        )
    if secondary_masses[0] < mass_min or secondary_masses[-1] > mass_max:
        raise ValueError(
            f"Secondary-mass grid lies outside POET's [{mass_min}, {mass_max}] Msun range"
        )
    if not feh_min <= base_system["feh"] <= feh_max:
        raise ValueError(
            f"[Fe/H]={base_system['feh']} lies outside POET's "
            f"[{feh_min}, {feh_max}] range"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser().resolve()
        if args.checkpoint_dir
        else output_dir / f"KIC{args.kic}_mass_eccentricity_grid_checkpoints"
    )
    signature = mass_grid_checkpoint_signature(
        args=args,
        base_system=base_system,
        data_dir=data_dir,
        eccentricity_database=eccentricity_database,
    )
    worker_args = argparse.Namespace(**vars(args))
    worker_args.lgq = [args.mass_grid_lgq]
    _MASS_GRID_RUNTIME.clear()
    _MASS_GRID_RUNTIME.update(
        args=worker_args,
        base_system=base_system,
        interpolator=interpolator,
        checkpoint_dir=checkpoint_dir,
        signature=signature,
        fixed_initial_period=fixed_initial_period,
    )
    _MASS_GRID_SPIN_CACHE.clear()

    trials: list[dict[str, Any]] = []
    pending: list[tuple[int, int, int, float, float, float]] = []
    for task in tasks:
        checkpoint_path = mass_grid_checkpoint_path(
            checkpoint_dir, task[3], task[4], task[5]
        )
        if args.resume and checkpoint_path.is_file():
            trial = load_mass_grid_checkpoint(
                checkpoint_path, task=task, signature=signature
            )
            if trial["success"] or not args.retry_failed:
                trials.append(trial)
                continue
        pending.append(task)

    resumed = len(trials)
    print(
        f"Checkpoints: {checkpoint_dir} ({resumed} resumed, "
        f"{len(pending)} pending)",
        flush=True,
    )
    if pending:
        pending_secondary_masses = sorted({task[4] for task in pending})
        print(
            "Precomputing secondary spin states before the evolution timer "
            f"({len(pending_secondary_masses)} unique M2 values)...",
            flush=True,
        )
        for secondary_mass in pending_secondary_masses:
            spin_system = dict(base_system)
            spin_system["secondary_mass"] = secondary_mass
            _MASS_GRID_SPIN_CACHE[round(secondary_mass, 12)] = (
                disk_locked_secondary_angmom(
                    spin_system,
                    interpolator,
                    worker_args,
                )
            )
    workers = min(args.num_parallel_processes, max(1, len(pending)))
    started = time.monotonic()

    def record(trial: dict[str, Any]) -> None:
        checkpoint_path = mass_grid_checkpoint_path(
            checkpoint_dir,
            trial["primary_mass"],
            trial["secondary_mass"],
            trial["initial_eccentricity"],
        )
        write_mass_grid_checkpoint(checkpoint_path, trial, signature)
        trials.append(trial)
        completed = len(trials)
        newly_completed = completed - resumed
        if (
            not trial["success"]
            or completed == total
            or newly_completed % args.mass_grid_progress_every == 0
        ):
            elapsed = time.monotonic() - started
            rate = newly_completed / elapsed if elapsed > 0.0 else 0.0
            eta = (len(pending) - newly_completed) / rate if rate > 0.0 else math.inf
            state = "ok" if trial["success"] else trial["status"]
            print(
                f"[{completed}/{total}] M1={trial['primary_mass']:.6g}, "
                f"M2={trial['secondary_mass']:.6g}, "
                f"e_i={trial['initial_eccentricity']:.6g}: {state}; "
                f"rate={rate:.3g}/s, ETA={eta / 60.0:.2f} min",
                flush=True,
            )

    if pending:
        if workers == 1:
            for task in pending:
                record(_run_mass_grid_trial(task))
        else:
            if "fork" not in mp.get_all_start_methods():
                raise RuntimeError(
                    "Parallel POET grid execution requires the multiprocessing "
                    "'fork' start method so workers inherit the shared native tables"
                )
            print(
                f"Running real POET on {workers} CPU worker processes. The native "
                "POET library has no CUDA backend.",
                flush=True,
            )
            context = mp.get_context("fork")
            tasks_per_child = args.mass_grid_tasks_per_child or None
            with context.Pool(
                processes=workers, maxtasksperchild=tasks_per_child
            ) as pool:
                for trial in pool.imap_unordered(
                    _run_mass_grid_trial,
                    pending,
                    # One point per work item provides real point-level
                    # concurrency and prevents a native failure from losing a
                    # whole mass-pair chunk. Secondary spin states are still
                    # cached inside each long-lived worker.
                    chunksize=1,
                ):
                    record(trial)

    successful = sum(trial["success"] for trial in trials)
    total_evolutions = sum(trial["num_period_evolutions"] for trial in trials)
    newly_timed = [
        trial
        for trial in trials
        if not trial.get("resumed", False)
        and math.isfinite(trial["evolution_seconds"])
        and math.isfinite(trial["evolution_started_at"])
        and math.isfinite(trial["evolution_finished_at"])
    ]
    if newly_timed:
        evolution_sum_seconds = sum(
            trial["evolution_seconds"] for trial in newly_timed
        )
        evolution_wall_seconds = max(
            trial["evolution_finished_at"] for trial in newly_timed
        ) - min(trial["evolution_started_at"] for trial in newly_timed)
        evolution_mean_seconds = evolution_sum_seconds / len(newly_timed)
    else:
        evolution_sum_seconds = float("nan")
        evolution_wall_seconds = float("nan")
        evolution_mean_seconds = float("nan")
    timing = {
        "num_systems_timed": len(newly_timed),
        "num_cpu_workers": workers,
        "parallel_evolution_wall_seconds": evolution_wall_seconds,
        "summed_per_system_evolution_seconds": evolution_sum_seconds,
        "mean_per_system_evolution_seconds": evolution_mean_seconds,
    }
    config = {
        **vars(args),
        "w19_data_dir": str(data_dir),
        "interpolator_dir": str(Path(args.interpolator_dir).expanduser().resolve()),
        "eccentricity_db": str(eccentricity_database),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_signature": signature,
        "base_system_medians": base_system,
        "primary_masses": primary_masses.tolist(),
        "secondary_masses": secondary_masses.tolist(),
        "initial_eccentricities": initial_eccentricities.tolist(),
        "fixed_initial_period_day": fixed_initial_period,
        "num_grid_trials": total,
        "num_successful_trials": successful,
        "num_failed_trials": total - successful,
        "num_poet_evolution_calls": total_evolutions,
        "execution_backend": "CPU process pool" if workers > 1 else "CPU serial",
        "cpu_evolution_timing": timing,
    }
    summary_path, arrays_path, config_path = save_mass_grid_results(
        output_dir=output_dir,
        kic=args.kic,
        primary_masses=primary_masses,
        secondary_masses=secondary_masses,
        eccentricities=initial_eccentricities,
        trials=trials,
        config=config,
    )
    print(
        f"Completed {successful}/{total} trials using {total_evolutions} POET "
        "evolution calls.",
        flush=True,
    )
    print("CPU EVOLUTION TIMING (POET evolution calls only)", flush=True)
    print(
        f"  Systems timed: {len(newly_timed)}; CPU workers: {workers}",
        flush=True,
    )
    print(
        "  Parallel wall time from first evolution start to last evolution "
        f"finish: {evolution_wall_seconds:.6f} s "
        f"({evolution_wall_seconds / 60.0:.6f} min)",
        flush=True,
    )
    print(
        "  Sum of individual evolution durations: "
        f"{evolution_sum_seconds:.6f} s",
        flush=True,
    )
    print(
        f"  Mean evolution duration per system: {evolution_mean_seconds:.6f} s",
        flush=True,
    )
    print(
        "  Excluded: eccentricity-table preload, interpolator setup, initial "
        "spin setup, checkpoint writes, and final output generation.",
        flush=True,
    )
    print(f"Mass-grid summary: {summary_path}", flush=True)
    print(f"Mass-grid endpoint arrays: {arrays_path}", flush=True)
    print(f"Mass-grid configuration: {config_path}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line options."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--kic", type=int, default=3348093)
    parser.add_argument(
        "--mass-eccentricity-grid",
        action="store_true",
        help=(
            "Run the parallel 3-D (primary mass, secondary mass, initial "
            "eccentricity) grid instead of the original lgQ/e diagnostic."
        ),
    )
    parser.add_argument(
        "--primary-mass-grid",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "COUNT"),
        help=(
            "Inclusive primary-mass grid in Msun. By default the selected "
            "W19 posterior mass quantiles are sampled at --mass-grid-size points."
        ),
    )
    parser.add_argument(
        "--secondary-mass-grid",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "COUNT"),
        help=(
            "Inclusive secondary-mass grid in Msun. By default the selected "
            "W19 posterior mass quantiles are sampled at --mass-grid-size points."
        ),
    )
    parser.add_argument(
        "--mass-grid-eccentricity",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "COUNT"),
        help=(
            "Inclusive initial-eccentricity grid. The default is 10 points "
            "from 0 through 0.8."
        ),
    )
    parser.add_argument(
        "--mass-grid-size",
        type=int,
        default=10,
        help="Default number of points along each of the three mass-grid axes.",
    )
    parser.add_argument(
        "--mass-quantiles",
        nargs=2,
        type=float,
        default=[0.05, 0.95],
        metavar=("LOW", "HIGH"),
        help="W19 posterior quantiles defining default primary/secondary mass ranges.",
    )
    parser.add_argument(
        "--mass-grid-lgq",
        type=float,
        default=6.5,
        help="Fixed log10(Q') used by every mass/eccentricity grid point.",
    )
    parser.add_argument(
        "--mass-grid-period-mode",
        choices=("fixed", "solve"),
        default="fixed",
        help=(
            "Use one fixed initial period (exactly one POET call per point), "
            "or solve for the initial period reproducing the observed period."
        ),
    )
    parser.add_argument(
        "--mass-grid-initial-period",
        type=float,
        default=None,
        help=(
            "Initial period in days for fixed period mode. Defaults to the "
            "selected system's median observed period."
        ),
    )
    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    parser.add_argument(
        "--num-parallel-processes",
        "--num-parallel",
        type=int,
        default=max(1, available_cpus // 2),
        help=(
            "Maximum number of concurrent CPU POET workers. Workers inherit "
            "the preloaded read-only eccentricity table using fork."
        ),
    )
    parser.add_argument(
        "--mass-grid-tasks-per-child",
        type=int,
        default=0,
        help="Recycle a worker after this many trials; zero keeps workers alive.",
    )
    parser.add_argument(
        "--mass-grid-progress-every",
        type=int,
        default=10,
        help="Report parallel progress after this many newly completed trials.",
    )
    parser.add_argument(
        "--mass-grid-dry-run",
        action="store_true",
        help="Print the derived 3-D grid without initializing or running POET.",
    )
    parser.add_argument(
        "--lgq",
        nargs="+",
        type=float,
        default=[
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.5,
            10.0,
            10.5,
            11.0,
            11.5,
            12.0,
        ],
        help="Values of log10(Q') to evolve; they need not be evenly spaced.",
    )
    parser.add_argument(
        "--initial-eccentricities",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help=(
            "Initial eccentricity quadrature grid for the original uniform "
            "prior on [0, 0.8]. The endpoints must be included for bounds."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Allow an incomplete one-point lgQ/e_i run for software testing. "
            "Its plot is explicitly marked 'bounds unavailable'."
        ),
    )
    parser.add_argument("--w19-data-dir", default=None)
    parser.add_argument(
        "--eccentricity-kde-cache",
        default=None,
        help=(
            "Cache file for the prior-corrected W19 eccentricity distribution. "
            "The default is a KIC-specific file under source/bayesian/cache."
        ),
    )
    parser.add_argument(
        "--interpolator-dir",
        default=str(POET_DIR / "stellar_evolution_interpolators"),
    )
    parser.add_argument("--eccentricity-db", default=None)
    parser.add_argument(
        "--preload-eccentricity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Preload eccentricity coefficients into RAM. This is strongly "
            "recommended for the default high-eccentricity scan."
        ),
    )
    parser.add_argument(
        "--mcmc-file",
        default=None,
        help="Optional emcee HDF5 file whose lgQ_min blobs override grid weights in the title.",
    )
    parser.add_argument("--mcmc-burn-in", type=int, default=0)
    parser.add_argument("--mcmc-thin", type=int, default=1)
    parser.add_argument("--disk-dissipation-age", type=float, default=5.0e-3)
    parser.add_argument("--disk-lock-period", type=float, default=5.0)
    parser.add_argument("--wind-strength", type=float, default=0.17)
    parser.add_argument("--wind-saturation", type=float, default=2.45)
    parser.add_argument("--core-envelope-coupling-timescale", type=float, default=1.0e-2)
    parser.add_argument("--max-time-step", type=float, default=1.0e-2)
    parser.add_argument("--precision", type=float, default=1.0e-6)
    parser.add_argument("--eccentricity-precision", type=float, default=1.0e-4)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output-samples", type=int, default=300)
    parser.add_argument("--period-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--period-search-factor", type=float, default=1.5)
    parser.add_argument("--min-initial-period", type=float, default=0.1)
    parser.add_argument("--max-initial-period", type=float, default=200.0)
    parser.add_argument("--max-bracket-steps", type=int, default=20)
    parser.add_argument("--max-root-iterations", type=int, default=50)
    parser.add_argument(
        "--confidence",
        type=float,
        default=float(normal_distribution.cdf(2.0) - normal_distribution.cdf(-2.0)),
        help="Central probability reported as the lgQ interval (default is Gaussian +/-2 sigma).",
    )
    parser.add_argument("--linear-age", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_PATH.parent / "single_system_lgq_output"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Directory containing one atomic NPZ checkpoint per grid point. "
            "Defaults to a KIC-specific directory below --output-dir."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse matching completed point checkpoints.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run checkpoints whose earlier trial ended in failure.",
    )
    parser.add_argument(
        "--recover-log",
        action="append",
        default=[],
        help=(
            "Earlier progress log from which to recover solved initial "
            "periods. May be passed more than once. Each recovered point is "
            "replayed once to reconstruct and checkpoint its trajectory."
        ),
    )
    args = parser.parse_args()

    args.lgq = sorted({float(value) for value in args.lgq})
    args.initial_eccentricities = sorted(
        {float(value) for value in args.initial_eccentricities}
    )
    if not args.lgq:
        parser.error("At least one --lgq value is required")
    if any(not 0.0 <= value <= 0.8 for value in args.initial_eccentricities):
        parser.error("Every initial eccentricity must be in [0, 0.8]")
    if args.mass_eccentricity_grid:
        if args.mass_grid_size < 1:
            parser.error("--mass-grid-size must be positive")
        quantile_low, quantile_high = args.mass_quantiles
        if not 0.0 <= quantile_low < quantile_high <= 1.0:
            parser.error("--mass-quantiles must satisfy 0 <= LOW < HIGH <= 1")
        if not math.isfinite(args.mass_grid_lgq):
            parser.error("--mass-grid-lgq must be finite")
        if args.mass_grid_initial_period is not None and args.mass_grid_initial_period <= 0.0:
            parser.error("--mass-grid-initial-period must be positive")
        if args.num_parallel_processes < 1:
            parser.error("--num-parallel-processes must be positive")
        if args.mass_grid_tasks_per_child < 0:
            parser.error("--mass-grid-tasks-per-child cannot be negative")
        if args.mass_grid_progress_every < 1:
            parser.error("--mass-grid-progress-every must be positive")
    elif not args.smoke_test:
        if len(args.lgq) < 3:
            parser.error(
                "At least three --lgq values are required to calculate bounds; "
                "use --smoke-test for a one-point software check"
            )
        if (
            len(args.initial_eccentricities) < 3
            or not math.isclose(args.initial_eccentricities[0], 0.0)
            or not math.isclose(args.initial_eccentricities[-1], 0.8)
        ):
            parser.error(
                "Bounds require at least three --initial-eccentricities "
                "spanning the full prior [0, 0.8]; use --smoke-test for an "
                "incomplete diagnostic"
            )
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must lie strictly between zero and one")
    if args.mcmc_burn_in < 0 or args.mcmc_thin < 1:
        parser.error("MCMC burn-in must be nonnegative and thinning must be positive")
    for name in (
        "disk_dissipation_age",
        "disk_lock_period",
        "core_envelope_coupling_timescale",
        "max_time_step",
        "precision",
        "timeout",
        "period_tolerance",
        "period_search_factor",
        "min_initial_period",
        "max_initial_period",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.period_search_factor <= 1.0:
        parser.error("--period-search-factor must be greater than one")
    if args.min_initial_period >= args.max_initial_period:
        parser.error("--min-initial-period must be below --max-initial-period")
    if args.output_samples < 2:
        parser.error("--output-samples must be at least two")
    return args


def main() -> None:
    """Run the requested one-system lgQ scan."""

    args = parse_args()
    lgq_values = np.asarray(args.lgq, dtype=float)
    initial_eccentricities = np.asarray(args.initial_eccentricities, dtype=float)
    data_dir = find_w19_data_dir(args.w19_data_dir)
    eccentricity_database = find_eccentricity_database(args.eccentricity_db)
    samples = load_w19_samples(data_dir, args.kic)
    system = representative_system(samples)
    if args.mass_eccentricity_grid:
        run_mass_eccentricity_grid(
            args=args,
            samples=samples,
            base_system=system,
            data_dir=data_dir,
            eccentricity_database=eccentricity_database,
        )
        return
    observed_eccentricity = samples["eccentricity"]

    # The paper likelihood does not use a naive one-dimensional KDE of |e|.
    # It reconstructs the (e sin w, e cos w) density and converts its implicit
    # prior to one uniform in eccentricity.  Point the original helper at the
    # full data directory selected above and use that exact distribution here.
    from bayesian import windemuth_et_al_util  # pylint: disable=import-outside-toplevel
    from bayesian.windemuth_eccentricity_distribution import (  # pylint: disable=import-outside-toplevel
        W19EccentricityDistribution,
    )

    windemuth_et_al_util._data_dir = str(data_dir)  # pylint: disable=protected-access
    legacy_eccentricity_cache = Path(
        "/home/vortebo/ctime/CircularizationDissipationConstraints/"
        "windemuth_eccentricity_distros_blah.pkl"
    )
    eccentricity_cache = (
        Path(args.eccentricity_kde_cache).expanduser().resolve()
        if args.eccentricity_kde_cache
        else (
            legacy_eccentricity_cache
            if args.kic == 3348093 and legacy_eccentricity_cache.is_file()
            else SCRIPT_PATH.parent
            / "cache"
            / f"KIC{args.kic}_w19_eccentricity.pkl"
        )
    )
    eccentricity_cache.parent.mkdir(parents=True, exist_ok=True)
    eccentricity_density = W19EccentricityDistribution(
        args.kic, pickle_fname=str(eccentricity_cache)
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser().resolve()
        if args.checkpoint_dir
        else output_dir / f"KIC{args.kic}_constant_q_scan_checkpoints"
    )
    recovery_logs = [Path(path).expanduser().resolve() for path in args.recover_log]
    missing_recovery_logs = [path for path in recovery_logs if not path.is_file()]
    if missing_recovery_logs:
        raise FileNotFoundError(
            "Recovery logs do not exist: "
            + ", ".join(str(path) for path in missing_recovery_logs)
        )
    checkpoint_signature = trial_checkpoint_signature(
        args=args,
        system=system,
        data_dir=data_dir,
        eccentricity_database=eccentricity_database,
    )
    recovered_periods = recovered_initial_periods(recovery_logs)

    print(f"W19 data: {data_dir}", flush=True)
    print(f"Eccentricity database: {eccentricity_database}", flush=True)
    print(f"Checkpoint directory: {checkpoint_dir}", flush=True)
    if recovery_logs:
        print(
            f"Recovered {len(recovered_periods)} solved initial periods from "
            + ", ".join(str(path) for path in recovery_logs),
            flush=True,
        )
    print(
        f"KIC {args.kic}: M1={system['primary_mass']:.6g}, "
        f"M2={system['secondary_mass']:.6g} Msun, "
        f"age={system['age_gyr']:.6g} Gyr, "
        f"P={system['orbital_period_day']:.9g} d, "
        f"e={system['eccentricity']:.6g}",
        flush=True,
    )

    print("Preparing POET eccentricity expansion...", flush=True)
    evolution_library.prepare_eccentricity_expansion(
        str(eccentricity_database).encode("ascii"),
        args.eccentricity_precision,
        args.preload_eccentricity,
        True,
    )
    manager = StellarEvolutionManager(str(Path(args.interpolator_dir).expanduser()))
    interpolator = manager.get_interpolator_by_name("default")
    MESAInterpolator.set_quantity_lower_limit("iconv", 1.0e-3)
    mass_min, mass_max = interpolator.mass_range()
    feh_min, feh_max = interpolator.feh_range()
    for component in ("primary_mass", "secondary_mass"):
        if not mass_min <= system[component] <= mass_max:
            raise ValueError(
                f"{component}={system[component]} lies outside POET's "
                f"[{mass_min}, {mass_max}] Msun range"
            )
    if not feh_min <= system["feh"] <= feh_max:
        raise ValueError(
            f"[Fe/H]={system['feh']} lies outside POET's [{feh_min}, {feh_max}] range"
        )

    print("Computing the secondary spin state at disk dispersal...", flush=True)
    secondary_initial_angmom = disk_locked_secondary_angmom(
        system, interpolator, args
    )
    trials: list[dict[str, Any]] = []
    total_trials = lgq_values.size * initial_eccentricities.size
    for lgq in lgq_values:
        for initial_eccentricity in initial_eccentricities:
            point_number = len(trials) + 1
            point = (float(lgq), float(initial_eccentricity))
            checkpoint_path = trial_checkpoint_path(checkpoint_dir, *point)
            if args.resume and checkpoint_path.is_file():
                trial = load_trial_checkpoint(
                    checkpoint_path,
                    lgq=point[0],
                    initial_eccentricity=point[1],
                    signature=checkpoint_signature,
                )
                if trial["success"] or not args.retry_failed:
                    trials.append(trial)
                    print(
                        f"[{point_number}/{total_trials}] RESUME "
                        f"lgQ={lgq:.6g}, initial e={initial_eccentricity:.6g} "
                        f"({trial['status']})",
                        flush=True,
                    )
                    continue
            print(
                f"[{point_number}/{total_trials}] "
                f"lgQ={lgq:.6g}, initial e={initial_eccentricity:.6g}",
                flush=True,
            )
            trial: dict[str, Any] = {
                "lgq": point[0],
                "initial_eccentricity": point[1],
                "success": False,
            }
            try:
                initial_period = recovered_periods.get(point)
                recovered = initial_period is not None
                if recovered:
                    print(
                        f"  replaying recovered P_i={initial_period:.8g} d "
                        "to restore its trajectory",
                        flush=True,
                    )
                    try:
                        trajectory = evolve_binary(
                            initial_period=initial_period,
                            initial_eccentricity=point[1],
                            lgq=point[0],
                            system=system,
                            interpolator=interpolator,
                            secondary_initial_angmom=secondary_initial_angmom,
                            args=args,
                            save_trajectory=True,
                        )
                        period_error = abs(
                            trajectory["final_period_day"]
                            - system["orbital_period_day"]
                        )
                        if (
                            not trajectory["reached_age"]
                            or period_error > 10.0 * args.period_tolerance
                        ):
                            raise RuntimeError(
                                "replayed endpoint does not reproduce the "
                                f"observed period (error={period_error:.6g} d)"
                            )
                        evaluations = 1
                    except Exception as recovery_error:
                        print(
                            f"  recovered period rejected: {recovery_error}; "
                            "solving again",
                            flush=True,
                        )
                        recovered = False
                if not recovered:
                    initial_period, trajectory, evaluations = solve_initial_period(
                        lgq=point[0],
                        initial_eccentricity=point[1],
                        system=system,
                        interpolator=interpolator,
                        secondary_initial_angmom=secondary_initial_angmom,
                        args=args,
                    )
                final_eccentricity = trajectory["final_eccentricity"]
                log_likelihood = float(
                    np.asarray(
                        eccentricity_density.logpdf(final_eccentricity)
                    ).reshape(-1)[0]
                )
                trial.update(
                    success=True,
                    status=("ok_recovered_from_log" if recovered else "ok"),
                    initial_period_day=initial_period,
                    final_period_day=trajectory["final_period_day"],
                    final_eccentricity=final_eccentricity,
                    eccentricity_log_likelihood=log_likelihood,
                    num_period_evolutions=evaluations,
                    num_solver_steps=trajectory["num_solver_steps"],
                    trajectory=trajectory,
                )
                print(
                    f"  solved P_i={initial_period:.8g} d; "
                    f"P_f={trajectory['final_period_day']:.8g} d, "
                    f"e_f={final_eccentricity:.8g}, log L_e={log_likelihood:.5g}",
                    flush=True,
                )
            except Exception as error:  # Keep the rest of a diagnostic grid running.
                trial.update(
                    status=f"{type(error).__name__}: {error}",
                    initial_period_day=float("nan"),
                    final_period_day=float("nan"),
                    final_eccentricity=float("nan"),
                    eccentricity_log_likelihood=float("-inf"),
                    num_period_evolutions=0,
                    num_solver_steps=0,
                )
                print(f"  FAILED: {trial['status']}", flush=True)
            write_trial_checkpoint(checkpoint_path, trial, checkpoint_signature)
            print(f"  checkpoint: {checkpoint_path}", flush=True)
            trials.append(trial)

    log_likelihood, probabilities = grid_lgq_distribution(
        lgq_values, initial_eccentricities, trials
    )
    posterior_samples: np.ndarray | None = None
    posterior_file: Path | None = None
    if args.mcmc_file:
        posterior_file = Path(args.mcmc_file).expanduser()
    else:
        automatic = Path(
            "/home/vortebo/ctime/CircularizationDissipationConstraints/samples/"
            f"{args.kic}_mcmc_powerlawlgQ_samples.h5"
        )
        if automatic.is_file():
            posterior_file = automatic
    posterior_samples, posterior_path = load_mcmc_lgq(
        posterior_file, args.mcmc_burn_in, args.mcmc_thin
    )
    if posterior_samples is None:
        posterior_source = "endpoint-weighted grid"
        if posterior_file is not None:
            print(
                f"No usable lgQ_min iterations in {posterior_file}; using grid weights.",
                flush=True,
            )
    else:
        posterior_source = "MCMC"
        print(
            f"Using {posterior_samples.size} lgQ_min samples from {posterior_path}",
            flush=True,
        )

    bounds = summarize_lgq_distribution(
        lgq_values=lgq_values,
        log_likelihood=log_likelihood,
        probabilities=probabilities,
        posterior_samples=posterior_samples,
        posterior_source=posterior_source,
        confidence=args.confidence,
        initial_eccentricities=initial_eccentricities,
    )
    if bounds["available"]:
        print(
            f"lgQ bounds ({100.0 * args.confidence:.2f}% central): "
            f"{bounds['lower_lgq']:.6g} < lgQ < {bounds['upper_lgq']:.6g}; "
            f"median={bounds['median_lgq']:.6g}",
            flush=True,
        )
        if (
            bounds["lower_grid_edge_limited"]
            or bounds["upper_grid_edge_limited"]
        ):
            print(
                "WARNING: At least one bound is grid-edge limited; extend the "
                "lgQ grid before interpreting it as a data-driven limit.",
                flush=True,
            )
    else:
        print(f"lgQ bounds unavailable: {bounds['note']}", flush=True)

    config = {
        **vars(args),
        "w19_data_dir": str(data_dir),
        "interpolator_dir": str(Path(args.interpolator_dir).expanduser().resolve()),
        "eccentricity_db": str(eccentricity_database),
        "eccentricity_kde_cache": str(eccentricity_cache),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_signature": checkpoint_signature,
        "recovered_initial_period_count": len(recovered_periods),
        "system_medians": system,
        "mcmc_source_used": posterior_path,
        "distribution_source": posterior_source,
        "lgq_bounds": bounds,
    }
    summary_path, track_path, bounds_path, config_path = save_results(
        output_dir=output_dir,
        kic=args.kic,
        trials=trials,
        lgq_values=lgq_values,
        log_likelihood=log_likelihood,
        probabilities=probabilities,
        bounds=bounds,
        config=config,
    )
    figure_path = plot_results(
        output_dir=output_dir,
        kic=args.kic,
        system=system,
        observed_eccentricity=observed_eccentricity,
        trials=trials,
        lgq_values=lgq_values,
        probabilities=probabilities,
        posterior_samples=posterior_samples,
        posterior_source=posterior_source,
        bounds=bounds,
        confidence=args.confidence,
        linear_age=args.linear_age,
    )
    print(f"Figure: {figure_path}", flush=True)
    print(f"Trial summary: {summary_path}", flush=True)
    print(f"Trajectories: {track_path}", flush=True)
    print(f"lgQ bounds: {bounds_path}", flush=True)
    print(f"Configuration: {config_path}", flush=True)


if __name__ == "__main__":
    main()
