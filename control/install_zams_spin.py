"""Install direct shared-epoch ZAMS spins in either independent Juno package.

Run with the package root as the sole argument. Existing files are backed up;
all patches are prepared and parsed before any file is changed. No evolution
or sampling is launched. Only explicit binary_start_age solves change behavior.
"""
import ast
from pathlib import Path
import sys


def once(text, old, new):
    if new in text:
        return text
    if text.count(old) != 1:
        raise ValueError('Missing or ambiguous patch anchor: ' + old[:100])
    return text.replace(old, new, 1)


def prepare(root):
    changes = {}
    name = 'source/bayesian/log_likelihood_windemuth_et_al.py'
    text = (root/name).read_text()
    old = "                    elif self.parameter_names_units[big_term][i][0] == 'secondary_disk_lock_period':\n                        continue"
    new = "                    elif self.parameter_names_units[big_term][i][0] == 'secondary_disk_lock_period':\n                        parameters['secondary_disk_period'] = self.get_parameter_value(encoded_parameters, 'secondary_disk_lock_period')\n                        continue"
    changes[name] = once(text, old, new)

    name = 'general_purpose_python_modules/solve_for_initial_values.py'
    text = (root/name).read_text()
    text = once(text, "secondary_disk_period=(secondary_disk_period\n                                   or\n                                   disk_period).to_value(units.day),", "secondary_disk_period=(disk_period if secondary_disk_period is None\n                                   else secondary_disk_period).to_value(units.day),")
    old = "        binary.configure(age=primary.core_formation_age(),"
    new = """        if self.explicit_binary_start:
            # Direct solid-body spins: L_zone = I_zone(t_start) * 2*pi/P.
            # This bypasses all disk-locked pre-main-sequence preparation.
            age = self.configuration['disk_dissipation_age']
            primary.select_interpolation_region(age)
            secondary.select_interpolation_region(age)
            omega = 2.0 * numpy.pi / self.target_state.Pdisk
            primary_angmom = numpy.array([
                primary.envelope_inertia(age) * omega,
                primary.core_inertia(age) * omega,
            ])
            spins = numpy.concatenate((primary_angmom, initial_secondary_angmom))
            if not numpy.isfinite(spins).all():
                raise ValueError('Nonfinite direct ZAMS angular momenta')
            binary.configure(age=age,
                             semimajor=binary.semimajor(porb_initial),
                             eccentricity=initial_eccentricity,
                             spin_angmom=spins,
                             inclination=numpy.zeros(4),
                             periapsis=numpy.zeros(3),
                             evolution_mode='BINARY')
            primary.detect_stellar_wind_saturation()
            secondary.detect_stellar_wind_saturation()
            return binary
        binary.configure(age=primary.core_formation_age(),"""
    text = once(text, old, new)
    old = "        secondary = self._create_secondary(False)\n\n        if not self.secondary_star:"
    new = """        secondary = self._create_secondary(self.explicit_binary_start)
        if self.explicit_binary_start and self.secondary_star:
            try:
                age = self.configuration['disk_dissipation_age']
                omega = 2.0 * numpy.pi / self.configuration['secondary_disk_period']
                result = (secondary.envelope_inertia(age) * omega,
                          secondary.core_inertia(age) * omega)
                if not numpy.isfinite(result).all():
                    raise ValueError('Nonfinite secondary ZAMS angular momenta')
                return result
            finally:
                secondary.delete()

        if not self.secondary_star:"""
    changes[name] = once(text, old, new)
    for name, text in changes.items():
        ast.parse(text, filename=name)
    return changes


def main():
    root = Path(sys.argv[1]).resolve()
    if root.name not in ('massive_star_circularization', 'gaia_massive_star_circularization'):
        raise ValueError('Expected an independent massive-star package root')
    changes = prepare(root)
    for name, text in changes.items():
        path = root/name
        backup = path.with_name(path.name + '.before_direct_zams_spin')
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        path.write_text(text)
    print('Installed direct ZAMS spins:', root)


if __name__ == '__main__':
    main()
