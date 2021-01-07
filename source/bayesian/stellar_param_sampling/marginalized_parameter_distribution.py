"""Defune the marginalized distribution of a single star parameter."""

from scipy.integrate import dblquad, tplquad

class MarginalizedParamterDistribution:
    """Marginalized distribution of one star variable from full 3-D distro."""

    def _scaled_3d_pdf(self, age, mass, feh):
        """The unnormalized 3-D PDF to marginalize."""

        if(
                self.limits['feh'][0] < feh < self.limits['feh'][1]
                and
                self.limits['mass'][0] < mass < self.limits['mass'][1]
                and
                (
                    self.limits['age'][0](feh, mass)
                    <
                    age
                    <
                    self.limits['age'][1](feh, mass)
                )
        ):
            return (self.direct_metallicity_distribution.pdf(feh)
                    *
                    self.conditional_mass_age_distribution([age, mass, feh]))

        return 0.0

    #pylint: disable=arguments-differ
    def pdf(self, x):
        """The PDF of the selected variable marginalized over the others."""

        x = float(x)
        print('Calculating PDF(%s) at %s = %s'
              %
              (self.variable, self.variable, x))
        if self.variable == 'feh':
            integrand = lambda age, mass: self._scaled_3d_pdf(age, mass, x)
            ylimits = (
                lambda mass: self.limits['age'][0](x, mass),
                lambda mass: self.limits['age'][1](x, mass)
            )

        elif self.variable == 'mass':
            integrand = lambda age, feh: self._scaled_3d_pdf(age, x, feh)
            ylimits = (
                lambda feh: self.limits['age'][0](feh, x),
                lambda feh: self.limits['age'][1](feh, x)
            )

        else:
            integrand = lambda mass, feh: self._scaled_3d_pdf(x, mass, feh)
            ylimits = self.limits['mass']

        integral = dblquad(
            integrand,
            *self.limits['mass' if self.variable == 'feh' else 'feh'],
            *ylimits,
            epsrel=1e-5
        )
        print('Result: ' + repr(integral))
        return integral[0] / self.normalization
    #pylint: enable=arguments-differ

    def __init__(self,
                 direct_metallicity_distribution,
                 conditional_mass_age_distribution,
                 variable,
                 limits,
                 *parent_args,
                 **parent_kwargs):
        """
        Set-up the distribution of one of the variables.

        Args:
            direct_metallicity_distribution:    The directly measured
                metellacity distribution. Will be constrained to lie within the
                [Fe/H] limits (see `limits` argument).

            conditional_mass_age_distribution:    Possibly unnormalized
                PDF(age, mass | [Fe/H]).

            variable:    Which variable to set-up the distribution for. Should
                be one of `'mass'`, `'age'`, or `'feh'`.

            limits:    Dictionary with keys `'mass'`, `'age'`, and `'feh'` each
                containing a 2-tuple giving the lower and upper limits for the
                corresponding variable.

            parent_args:    Passed directly to parent's :meth:`__init__`.

            parent_kwargs:    Passed directly to parent's :meth:`__init__`.
        """

        self.direct_metallicity_distribution = direct_metallicity_distribution
        self.conditional_mass_age_distribution = (
            conditional_mass_age_distribution
        )
        self.variable = variable
        self.limits = limits

        print('Calculating normalization')
        self.normalization = tplquad(
            self._scaled_3d_pdf,
            *limits['feh'],
            *limits['mass'],
            *limits['age'],
            epsrel=1e-5
        )
        print('Normalization = ' + repr(self.normalization))
        self.normalization = self.normalization[0]
