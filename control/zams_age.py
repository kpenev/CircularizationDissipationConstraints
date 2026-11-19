"""Independent bilinear lookup of the primary's EEP-202 age.

Mass and [Fe/H] coordinates are linear; the interpolated value is log(age).
No SciPy, POET, or native stellar-interpolation imports are needed.
"""
import argparse
from bisect import bisect_left
import csv
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path

POLICY = 'primary_zams_bilinear'
SCHEMA = 'primary-zams-bilinear-v1'
CONFIGURATION = 'kartof-mist2-three-zone-v1'
DEFAULT_ROOT = Path('/work/dal205149/high-mass-binaries-mist2')


def default_grid_path():
    root = Path(os.environ.get('HIGH_MASS_RUN_ROOT', DEFAULT_ROOT))
    return Path(os.environ.get('ZAMS_AGE_GRID', root / 'zams_age_grid.json'))


def _bracket(axis, value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{label} must be finite')
    index = bisect_left(axis, value)
    for candidate in (index-1, index):
        if 0 <= candidate < len(axis) and math.isclose(value, axis[candidate], rel_tol=0, abs_tol=1e-12):
            return [(candidate, 1.0)]
    if index == 0 or index == len(axis):
        raise ValueError(f'{label}={value} outside [{axis[0]}, {axis[-1]}]; extrapolation is disabled')
    weight = (value-axis[index-1])/(axis[index]-axis[index-1])
    return [(index-1, 1-weight), (index, weight)]


class ZamsAgeGrid:
    def __init__(self, document, *, grid_path=None, grid_sha256=None):
        if document.get('schema') != SCHEMA or document.get('eep') != 202:
            raise ValueError('Expected an EEP-202 bilinear ZAMS-age grid')
        if document.get('source_configuration') != CONFIGURATION:
            raise ValueError('Expected the new Kartof-matched MIST2 grid')
        if document.get('interpolation') != 'bilinear in mass and FeH; natural logarithm of age in years':
            raise ValueError('Unexpected ZAMS interpolation convention')
        self.masses = tuple(document['masses_msun'])
        self.fehs = tuple(document['fehs'])
        expected_m = tuple(i/100 for i in range(150, 301, 5))
        allowed_f = [tuple(i/100 for i in range(low,71,5)) for low in (-110,-120)]
        if self.masses != expected_m or self.fehs not in allowed_f:
            raise ValueError('Expected the complete original or extended massive-star grid')
        self.ages = tuple(tuple(row) for row in document['zams_age_yr'])
        if len(self.ages) != len(self.masses) or any(len(row) != len(self.fehs) for row in self.ages):
            raise ValueError('Incomplete ZAMS-age array')
        if any(not math.isfinite(age) or age <= 0 for row in self.ages for age in row):
            raise ValueError('ZAMS ages must be positive and finite')
        self.grid_path = str(grid_path) if grid_path is not None else None
        self.grid_sha256 = grid_sha256

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        content = path.read_bytes()
        return cls(json.loads(content), grid_path=path,
                   grid_sha256=hashlib.sha256(content).hexdigest())

    def evaluate(self, primary_mass, feh):
        mass_nodes = _bracket(self.masses, primary_mass, 'primary mass (Msun)')
        feh_nodes = _bracket(self.fehs, feh, '[Fe/H]')
        corners = [dict(mass_msun=self.masses[i], feh=self.fehs[j],
                        zams_age_yr=self.ages[i][j], weight=wm*wf)
                   for i, wm in mass_nodes for j, wf in feh_nodes]
        # Return exact grid-node values without a logarithm round trip.
        age = (corners[0]['zams_age_yr'] if len(corners) == 1 else
               math.exp(math.fsum(c['weight']*math.log(c['zams_age_yr']) for c in corners)))
        return dict(policy=POLICY, primary_mass_msun=float(primary_mass), feh=float(feh),
                    starting_age_gyr=age/1e9, starting_age_myr=age/1e6,
                    zams_eep=202, source_configuration=CONFIGURATION,
                    method='bilinear in mass and FeH; natural logarithm of age in years',
                    grid_path=self.grid_path, grid_sha256=self.grid_sha256,
                    corners=corners)

    def binary_start(self, primary_mass, feh, *, secondary_mass=None,
                     final_age_gyr=None, support_bounds=None):
        """Use the primary's ZAMS as one shared age, including for the secondary.

        support_bounds(mass, feh) must return the intersection of the actual
        stellar quantities' supported age intervals, in Gyr. If omitted, this
        method validates mass/FeH coverage only, and reports that explicitly.
        """
        result = self.evaluate(primary_mass, feh)
        age = result['starting_age_gyr']
        components = [('primary', float(primary_mass))]
        if secondary_mass is not None:
            _bracket(self.masses, secondary_mass, 'secondary mass (Msun)')
            components.append(('secondary', float(secondary_mass)))
            result['secondary_mass_msun'] = float(secondary_mass)
        if final_age_gyr is not None:
            final_age_gyr = float(final_age_gyr)
            if not math.isfinite(final_age_gyr) or not age < final_age_gyr:
                raise ValueError('Final age must be finite and later than the primary ZAMS')
            result['final_age_gyr'] = final_age_gyr
        result['stellar_support_checked'] = support_bounds is not None
        if support_bounds is not None:
            intervals = {}
            for name, mass in components:
                lower, upper = map(float, support_bounds(mass, float(feh)))
                if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
                    raise ValueError(f'Invalid {name} stellar age support')
                if not lower < age < upper:
                    raise ValueError(f'Primary ZAMS age {age} Gyr is outside {name} stellar support ({lower}, {upper})')
                if final_age_gyr is not None and final_age_gyr > upper:
                    raise ValueError(f'Requested final age is outside {name} stellar support')
                intervals[name] = dict(min_age_gyr=lower, max_age_gyr=upper)
            result['stellar_support'] = intervals
        return result


@lru_cache(maxsize=8)
def _cached_grid(path):
    return ZamsAgeGrid.load(path)


def primary_zams_start(primary_mass, feh, *, grid_path=None, **kwargs):
    """Reusable start-age lookup; cached once per process for inference calls."""
    path = str(Path(grid_path if grid_path is not None else default_grid_path()).resolve())
    return _cached_grid(path).binary_start(primary_mass, feh, **kwargs)


def zams_age_gyr(primary_mass, feh, *, grid_path=None):
    return primary_zams_start(primary_mass, feh, grid_path=grid_path)['starting_age_gyr']


def build_grid(ages_csv, output, models_root):
    """Package previously validated EEP milestone ages; copy no stellar tracks."""
    ages_csv, output = Path(ages_csv), Path(output)
    models_root = Path(models_root)
    with ages_csv.open() as stream:
        rows = list(csv.DictReader(stream))
    lookup = {}
    for row in rows:
        key = (float(row['mass_msun']), float(row['feh']))
        zams, tams = float(row['zams_age_yr']), float(row['tams_age_yr'])
        source = Path(row['source_track'])
        if (key in lookup or int(row['zams_eep']) != 202 or
                not math.isfinite(tams) or not 0 < zams < tams or
                not source.is_relative_to(models_root)):
            raise ValueError(f'Duplicate, invalid or incorrect-source milestone: {row["case_id"]}')
        lookup[key] = zams
    masses = [i/100 for i in range(150, 301, 5)]
    fehs = sorted({f for m,f in lookup})
    if fehs not in [[i/100 for i in range(low,71,5)] for low in (-110,-120)]:
        raise ValueError('Unexpected metallicity range in ZAMS milestone table')
    if set(lookup) != {(m, f) for m in masses for f in fehs}:
        raise ValueError('Need the complete MIST2 ZAMS age rectangle')
    document = dict(schema=SCHEMA, eep=202, source_configuration=CONFIGURATION,
                    interpolation='bilinear in mass and FeH; natural logarithm of age in years',
                    source_models_root=str(models_root), source_csv=str(ages_csv),
                    source_csv_sha256=hashlib.sha256(ages_csv.read_bytes()).hexdigest(),
                    masses_msun=masses, fehs=fehs,
                    zams_age_yr=[[lookup[m, f] for f in fehs] for m in masses])
    ZamsAgeGrid(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    temporary.write_text(json.dumps(document, indent=2)+'\n')
    temporary.replace(output)
    return dict(grid=str(output), count=len(lookup), eep=202,
                sha256=hashlib.sha256(output.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    build = sub.add_parser('build')
    build.add_argument('--ages', type=Path, required=True)
    build.add_argument('--output', type=Path, default=default_grid_path())
    build.add_argument('--models-root', type=Path, default=Path('/work/dal205149/mesa-grid-mist2/models'))
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--primary-mass', type=float, required=True)
    evaluate.add_argument('--feh', type=float, required=True)
    evaluate.add_argument('--secondary-mass', type=float)
    evaluate.add_argument('--final-age-gyr', type=float)
    evaluate.add_argument('--grid', type=Path, default=default_grid_path())
    args = parser.parse_args()
    if args.action == 'build':
        result = build_grid(args.ages, args.output, args.models_root)
    else:
        result = primary_zams_start(args.primary_mass, args.feh, grid_path=args.grid,
                                    secondary_mass=args.secondary_mass,
                                    final_age_gyr=args.final_age_gyr)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
