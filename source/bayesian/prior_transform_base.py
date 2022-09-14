"""Define base class for transforming the unit cube to evolution parameters."""

from abc import ABCMeta, abstractmethod
import logging
import itertools

import numpy

#The foal is to define simple callable.
#pylint: disable=too-few-public-methods
class PriorTransformBase(metaclass=ABCMeta):
    """
    Base class for transforming the unit cube to evolution parameters.

    Attrs:
        independent_parameter_distributions(iterable of 3-tuples):     The
            names, distributions and units of the directly observable parameters
            we will sample directly from. The distributions must provide the
            `scipy.stats.continuous_rv` interface. For parameters for which
            only a fixed values should be assumed, the distribution must be
            specified as a single numeric value convertible to float. Fixed
            value parameters do not consume unit cube entries. Finally, if
            the distribution is None, the parameter is assumed to follow
            uniform(0, 1) distribution with the specified units.

        parameter_order:    The full list of model parameter names and their
            corresponding units the transformation must produce in the order in
            which they will appear in the output array.
    """

    _logger = logging.getLogger(__name__)

    _priors_order = ['dissipation', 'evolution', 'system']

    def _fill_independent_parameters(self,
                                     unit_cube_iter,
                                     model_parameters,
                                     identify=False):
        """
        Consume unit-cube values to fill independently distributed parameters.

        Args:
            unit_cube_iter(iter):    Iterator over the unit cube values. Values
                get consumed for all non-fixed independent parameters.

            model_parameters(dict):    Updated with the values (and units) of
                the independent model parameters corresponding to the unit cube.
                If None, just advances the unit cube iterator without any
                calculations. The latter is used to count the number of free
                parameters required.

            identify(bool):    If True, model_parameters are filled with the
                value in `unit_cube_iter` corresponding to each parameter rather
                than applying the prior transform. Intended to allow identifying
                which unit cube entry corresponds to which parameter.

        Returns:
            None
        """

        for (
                name,
                distribution,
                param_units
        ) in (
            self.independent_parameter_distributions
        ):
            if distribution is None or identify:
                if model_parameters is None:
                    next(unit_cube_iter)
                else:
                    model_parameters[name] = next(unit_cube_iter) * param_units
            elif hasattr(distribution, 'ppf'):
                if model_parameters is None:
                    next(unit_cube_iter)
                else:
                    model_parameters[name] = (
                        distribution.ppf(next(unit_cube_iter))
                        *
                        param_units
                    )
            elif model_parameters is not None:
                try:
                    model_parameters[name] = float(distribution) * param_units
                except TypeError:
                    raise TypeError(
                        'Invalid direct observable %s = %s * %s! Should be '
                        'scipy.stats distribution, None, or numeric.'
                        %
                        (repr(name), repr(distribution), repr(param_units))
                    ) from None

    @abstractmethod
    def _fill_coupled_parameters(self,
                                 unit_cube_iter,
                                 model_parameters,
                                 identify=False):
        """
        Update input with parameters not distributed independntly of all others.

        Args:
            unit_cube_iter:    Iterator over the unit cube values which were not
                consumed by :meth:`_fill_direct_observables()`. Values are
                consumed to as needed.

            model_parameters:    The independtly distributed parameters already
                filled by :meth:`_fill_independent_parameters()`. Gets updated
                with the dependent parameters.

            identify(bool):    See same argument of
                :meth:`_fill_independent_parameters()`

        Returns:
            None
        """

    def __init__(self,
                 *,
                 independent_parameter_distributions,
                 model_parameter_order):
        """
        Set-up the prior transform per the given system and evolution params.

        Args:
            independent_parameter_distributions:    See same name attribute.

            model_parameter_order([2-tuples]):    See :attr:`parameter_order`.

        Returns:
            None
        """

        self.independent_parameter_distributions = (
            independent_parameter_distributions
        )
        self.parameter_order = model_parameter_order

    def __call__(self, unit_cube_values):
        """Return an array of the parameter values for evolving the system."""

        transformed_values = numpy.full(shape=(len(self.parameter_order),),
                                        fill_value=numpy.nan,
                                        dtype=float)

        model_parameters = dict()
        unit_cube_iter = iter(unit_cube_values)
        try:
            self._fill_independent_parameters(unit_cube_iter, model_parameters)
            self._fill_coupled_parameters(unit_cube_iter, model_parameters)
        except StopIteration:
            raise IndexError('Too few unit cube values provided to generate '
                             'model parameters!') from None

        try:
            next(unit_cube_iter)
        except StopIteration:
            pass
        else:
            raise IndexError('Too many unit cube values provided for '
                             'generating model parameters!') from None

        for param_index, (param_name, param_units) in enumerate(
                self.parameter_order
        ):
            if param_name in model_parameters:
                transformed_values[param_index] = (
                    model_parameters[param_name].to_value(param_units)
                )
            else:
                try:
                    transformed_values[param_index] = getattr(
                        self,
                        '_calculate_' + param_name
                    )(
                        model_parameters
                    ).to_value(
                        param_units
                    )
                except AttributeError:
                    raise RuntimeError(
                        'No method for calculating %s parameter found!'
                        %
                        repr(param_name)
                    ) from None

        self._logger.debug(
            'Prior transform: U(%s) -> Parameters:\n\t%s',
            repr(unit_cube_values),
            '\n\t'.join(
                '%s: %s' % (param, repr(value))
                for (param, _), value in zip(self.parameter_order,
                                             transformed_values)
            )
        )

        return dict(parameters=transformed_values)


    def count_sampled_parameters(self):
        """Count the random variates required by the defined transform."""

        counter = itertools.count()
        self._fill_independent_parameters(counter, None)
        self._fill_coupled_parameters(counter, None)
        return next(counter)


    def get_unit_cube_indices(self):
        """
        Return index of unit cube that determines value of each parameter.

        Args:
            None

        Returns:
            dict:
                Keys are the names of the parameters that are being sampled, and
                the entry is the index within the unit cube that directly
                determines the value of the corresponding parameter (given
                values of all prior parameters).
        """

        result = dict()
        counter = itertools.count()
        self._fill_independent_parameters(counter, result, True)
        self._fill_coupled_parameters(counter, result, True)
        return result
#pylint: enable=too-few-public-methods
