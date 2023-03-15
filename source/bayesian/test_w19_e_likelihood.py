"""Script to test how well W19 likelihood is approximated by f(e_final) = U."""

import numpy
from scipy import stats

from general_purpose_python_modules import KDEDistribution

from bayesian.windemuth_et_al_util import \
    get_samples,\
    get_available_kic
from bayesian.final_initial_eccentricity_dependence import\
    parse_command_line,\
    calculate_efinal

class EccentricityKernel(stats.rv_continuous):

def get_eccentricity_distro(kic_id):
    """Return the distribution of observed eccentricity for given KIC."""

    samples = get_samples(kic_id)
    samples.insert(
        loc=samples.shape[1],
        column='e',
        value=numpy.sqrt(
            samples['esinw']**2
            +
            samples['ecosw']**2
        )
    )
    eccentricity_kernel = (
        'rdist',
        (),
        dict(
            c=4,
            scale=max(
                (
                    numpy.std(samples['e'])
                    *
                    samples['e'].size**(-0.2)
                ),
                0.0001
            )
        )
    )

    return KDEDistribution(samples['e'], eccentricity_kernel)


def get_approximation_precision(e_distro, e_final, e_initial):
    """Return fractional error in likelihood if ef(ei) is assumed linear."""

    <++> Add 1/e to distribution. <++>
    interp_likelihood = 0.0
    cdf_bot = 0.0
    for segment in range(len(e_final) - 1):
        inverse_derivative = (
            (e_initial[segment + 1] - e_initial[segment])
            /
            (e_final[segment + 1] - e_final[segment])
        )
        cdf_top = e_distro.cdf(e_final[segment + 1])
        interp_likelihood += inverse_derivative * (cdf_top - cdf_bot)
        cdf_bot = cdf_top
    interp_likelihood /= e_initial[-1]

    linear_likelihood = cdf_top / e_final[-1]

    return (linear_likelihood - interp_likelihood) / interp_likelihood


def get_all_precisions(kic_id, max_e_final_grid, e_initial_grid, e_final_list):
    """Calculate fractional likelihood errors for all e_final."""

    e_distro = get_eccentricity_distro(kic_id)
    return [
        get_approximation_precision(
            e_distro,
            e_final_list[max_ef_ind * len(e_initial_grid)
                         :
                         (max_ef_ind + 1) * len(e_initial_grid)],
            e_initial_grid
        )
        for max_ef_ind in range(len(max_e_final_grid))
    ]


def main(config):
    """Avoid polluting global namespace."""

    w19_systems = get_available_kic()
    e_final_list = calculate_efinal(config)[1]
    print(('#25s' + ' %25.16f' * len(config.max_efinal_grid))
          %
          (('KIC',) + tuple(config.max_efinal_grid)))
    for kic_id in w19_systems:
        precisions = get_all_precisions(kic_id,
                                        config.max_efinal_grid,
                                        config.initial_eccentricity_grid,
                                        e_final_list)
        print(('#25d' + ' %25.16f' * len(config.max_efinal_grid))
              %
              ((kic_id,) + tuple(precisions)))


if __name__ == '__main__':
    main(parse_command_line)
