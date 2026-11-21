"""Selection, cache provenance and shared-ZAMS support for the massive profile."""
from functools import lru_cache
import json
import math
import os
from pathlib import Path

from zams_age import ZamsAgeGrid

PROFILE = 'massive-rc-env-primary-zams-v1'
MASS_RANGE = (1.5, 3.0)
FEH_RANGE = (-1.2, .7)
LOW_MASS_MAX = 1.65
FLOORS = dict(ICONV=1e-3, IRAD=1e-4, MRAD=1e-3, RRAD=0.)
CACHE_ROOTS = (
    Path('/work/dal205149/high-mass-binaries-mist2-feh-m1p20'),
    Path('/work/dal205149/high-mass-binaries-mist2'),
)


def validate_selection(primary_mass, secondary_mass, feh):
    values = tuple(map(float, (primary_mass, secondary_mass, feh)))
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Masses and [Fe/H] must be finite')
    m1, m2, feh = values
    if not all(MASS_RANGE[0] <= m <= MASS_RANGE[1] for m in (m1, m2)):
        raise ValueError('Both component masses must lie within 1.50–3.00 Msun')
    if not FEH_RANGE[0] <= feh <= FEH_RANGE[1]:
        raise ValueError('[Fe/H] must lie within -1.20 to +0.70')
    # Final user-selected Kepler sample has no mass-dependent metallicity cut.
    return values


@lru_cache(maxsize=4)
def validate_cache_root(root):
    root = Path(root).resolve()
    build = json.loads((root/'BUILD_DONE.json').read_text())
    validation = json.loads((root/'zams_age_validation.json').read_text())
    grid = ZamsAgeGrid.load(root/'zams_age_grid.json')
    if build.get('configuration') != 'kartof-mist2-three-zone-v1':
        raise ValueError('The selected cache is not the generated MIST2 grid')
    count = len(grid.masses)*len(grid.fehs)
    if build.get('tracks_per_mode') != count or validation.get('source_nodes_checked') != count:
        raise ValueError('Native and ZAMS grid counts do not match')
    if validation.get('grid_sha256') != grid.grid_sha256:
        raise ValueError('The selected ZAMS grid has not passed its recorded validation')
    if validation.get('native_cases_checked', 0) < 1:
        raise ValueError('Native support validation is missing')
    if not (root/'interpolators/rc_env/serialized.sqlite').is_file():
        raise ValueError('Missing rc_env native interpolator')
    return root


def select_cache_root():
    explicit = os.environ.get('MASSIVE_STAR_GRID_ROOT')
    if explicit:
        return validate_cache_root(explicit)
    failures = []
    for root in CACHE_ROOTS:
        try:
            return validate_cache_root(root)
        except (OSError, ValueError, KeyError) as exc:
            failures.append(f'{root}: {exc}')
    raise ValueError('No completed, validated massive-star cache: '+'; '.join(failures))


@lru_cache(maxsize=4)
def zams_grid(root):
    return ZamsAgeGrid.load(Path(root)/'zams_age_grid.json')


def setup_interpolator_config(config):
    directory = Path(config.stellar_evolution_interpolator_dir).resolve()
    if directory.name != 'rc_env' or directory.parent.name != 'interpolators':
        raise ValueError('This profile requires the generated rc_env interpolator directory')
    root = validate_cache_root(directory.parent.parent)
    os.environ['MASSIVE_STAR_GRID_ROOT'] = str(root)
    return root


def prepare_binary_start(interpolator, primary_mass, secondary_mass, feh,
                         final_age_gyr, *, grid_root=None):
    m1, m2, feh = validate_selection(primary_mass, secondary_mass, feh)
    # Check available cache coordinates before calling native quantity creation.
    mass_min, mass_max = map(float, interpolator.mass_range())
    feh_min, feh_max = map(float, interpolator.feh_range())
    if not all(mass_min <= m <= mass_max for m in (m1,m2)) or not feh_min <= feh <= feh_max:
        raise ValueError('Proposal is outside the currently loaded native grid')
    root = str(Path(grid_root) if grid_root is not None else select_cache_root())

    def bounds(mass, metallicity):
        intervals = []
        for name in interpolator.quantity_list:
            quantity = interpolator(name, mass, metallicity)
            try:
                intervals.append((float(quantity.min_age),float(quantity.max_age)))
            finally:
                quantity.delete()
        return max(x[0] for x in intervals), min(x[1] for x in intervals)

    result = zams_grid(root).binary_start(m1,feh,secondary_mass=m2,
                                         final_age_gyr=final_age_gyr,support_bounds=bounds)
    start = result['starting_age_gyr']
    for mass in set((m1,m2)):
        for name in interpolator.quantity_list:
            quantity = interpolator(name,mass,feh)
            try:
                if not all(math.isfinite(float(quantity(age))) for age in (start,float(final_age_gyr))):
                    raise ValueError('Nonfinite '+name+' at the requested start/end')
                if not all(math.isfinite(float(x)) for x in quantity.deriv(start)):
                    raise ValueError('Nonfinite '+name+' derivative at the starting age')
            finally:
                quantity.delete()
    return result
