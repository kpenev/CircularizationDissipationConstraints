"""Define class using machine learning for faster initial condition solving."""

from os import path
import logging
from multiprocessing import Lock

import numpy

from orbital_evolution import \
    InitialConditionSolver,\
    LockedPlanet,\
    lgQ


class MockLearnInitialConditions:
    """Provide I.C. machine learning interface, only collecting traning data."""

    _locks = dict()

    def __init__(self, model_id, reset=False):
        """
        Open or create new ML model for initial conditions.

        Args:
            model_id(str):    Unique identifier to assign to the model. If a
                model with this ID already existts, it is opened for querries
                and further training.

            reset(bool):    If true, if a model with the given ID exists, it is
                deleted, starting a new one.

        Returns:
            None
        """

        self.model_id = model_id
        self._fname = self.model_id + '.npz'
        self._locks[model_id] = Lock()
        if reset or not path.exists(self._fname):
            open(self._fname, 'wb').close()


    def add_training_data(self, ml_features, initial_value):
        """Add a set of features for which the initial condition is known."""

        with self._locks[self.model_id]:
            logging.getLogger(__name__).debug(
                'Saving training data to %s: \n%s\n->\n%s',
                repr(self._fname),
                repr(ml_features),
                repr(initial_value)
            )
            with open(self._fname, 'ab') as training_f:
                numpy.save(training_f, ml_features)
                numpy.save(training_f, initial_value)



    def __call__(self, ml_features):
        """
        Return best guess for inital Porb range given system parameters.

        Return ``None, None`` if model is not yet trained.
        """

        return None, None


class LearningICSolver(InitialConditionSolver):
    """Use machine learning wrapped around I.C. solver to speed up solution."""

    #pylint: disable=too-many-branches
    #pylint: disable=too-many-statements
    def _get_porb_solver_ml_features(self, disk_period, final_porb=None):
        """
        Return numpy array of values required by the ML to guess Porb range.

        Args:
            disk_period(float):    The disk period the evolution will be
                starting with.

            final_porb(float):    The final orbital period that must be
                reproduced. By default ``self.target.Porb`` is used.

        Returns:
            numpy.array:
                The values of the features machine learning uses to predict the
                initial orbital period range. The first value is always the
                final orbital period.
        """

        result = numpy.empty(len(self._learning_features))
        ind = 0
        for feature in self._learning_features:
            if feature == 'final porb':
                result[ind] = final_porb
            elif feature == 'm1':
                result[ind] = self.primary.mass
            elif feature == 'm2':
                result[ind] = self.secondary.mass
            elif feature == 'r2' and isinstance(self.secondary, LockedPlanet):
                result[ind] = self.secondary.radius
            elif feature == 'age':
                result[ind] = self.target.age
            elif feature == 'feh':
                assert self.primary.metallicity == self.secondary.metallicity
                result[ind] = self.primary.metallicity
            elif feature == 'Kwind1':
                result[ind] = self.primary.wind_strength
            elif feature == 'Kwind2':
                result[ind] = self.secondary.wind_strength
            elif feature == 'wsat1':
                result[ind] = self.primary.wind_saturation_frequency
            elif feature == 'wsat2':
                result[ind] = self.secondary.wind_saturation_frequency
            elif feature ==  'diff rot coupling timescale1':
                result[ind] = self.primary.diff_rot_coupling_timescale
            elif feature ==  'diff rot coupling timescale2':
                result[ind] = self.secondary.diff_rot_coupling_timescale
            elif feature == 'Pdisk':
                result[ind] = disk_period
            elif feature == 'i0':
                result[ind] = self.parameters['initial inclination']
            elif feature == 'e0':
                result[ind] = self.parameters['initial eccentricity']
            elif feature ==  'L0':
                result[ind] = self.parameters['initial secondary angmom'][0]
                ind += 1
                result[ind] = self.parameters['initial secondary angmom'][1]
            elif feature == 'formation age':
                result[ind] = self.target.planet_formation_age
            elif feature == 'tdisk':
                result[ind] = self.target.disk_dissipation_age
            elif feature in ['lgQ1',
                             'satpowerlaw_lgQ1',
                             'lgQ2',
                             'satpowerlaw_lgQ2']:
                dissipation = (
                    self.primary if feature[-1] == '1' else self.secondary
                ).dissipation[0]
                result[ind] = lgQ(
                    dissipation['reference_phase_lag']
                )
                if feature == 'satpowerlaw_lgQ1':
                    assert len(dissipation['tidal_frequency_breaks']) == 1
                    ind += 1
                    assert not (
                        dissipation['tidal_frequency_powers'][0]
                        and
                        dissipation['tidal_frequency_powers'][1]
                    )
                    result[ind] = (
                        dissipation['tidal_frequency_powers'][0]
                        or
                        dissipation['tidal_frequency_powers'][1]
                    )
                    ind += 1
                    result[ind] = dissipation['tidal_frequency_breaks'][0]

            ind += 1

        return result
    #pylint: enable=too-many-branches
    #pylint: enable=too-many-statements


    def _try_initial_conditions(self,
                                initial_orbital_period,
                                disk_period,
                                save=False):
        """
        Pass training data to machine learning model for initial conditions.

        See `InitialConditionSolver._try_initial_conditions()` for full
        documentation.
        """

        logging.getLogger(__name__).debug(
            'Trying initial conditions initial Porb = %s, Pdisk = %s.',
            repr(initial_orbital_period),
            repr(disk_period)
        )

        final_porb, final_primary_pspin = super()._try_initial_conditions(
            initial_orbital_period,
            disk_period,
            save
        )
        if final_porb:
            ml_features = self._get_porb_solver_ml_features(disk_period,
                                                            final_porb)
            if self.validate_training_data(ml_features, initial_orbital_period):
                self._guess_porb_range.add_training_data(
                    ml_features,
                    initial_orbital_period
                )
        return final_porb, final_primary_pspin


    def _find_porb_range(self, guess_porb_initial, disk_period):
        """
        ML aided finding of period range where final porb error flips sign.

        See `InitialConditionSolver._find_porb_range()` for full description.

        If the machine learning produces estimated period range,
        ``guess_porb_initial`` is ignored.
        """

        ml_porb_range = self._guess_porb_range(
            self._get_porb_solver_ml_features(disk_period)
        )
        if ml_porb_range[0] is None:
            return super()._find_porb_range(guess_porb_initial, disk_period)

        assert ml_porb_range[1] > ml_porb_range[0]
        period_search_factor = self.period_search_factor
        self.period_search_factor = ml_porb_range[1] / ml_porb_range[0]
        result = super()._find_porb_range(ml_porb_range[0], disk_period)
        self.period_search_factor = period_search_factor
        return result


    def __init__(self,
                 *,
                 ic_model_id='default_ic_model',
                 learning_features,
                 reset_ic_model=False,
                 **kwargs):
        """
        Prepare the solver.

        Args:
            ic_model_id(str):    See :meth:`MockLearnInitialConditions.__init__`

            reset_ic_model(bool):    See
                :meth:`MockLearnInitialConditions.__init__`

            learning_features(iterable):    List of properties to include in the
                features that are used for machine learning. Should be a subset
                of:

                    * 'final porb': final orbital period

                    * 'm1': primary mass

                    * 'm2': secondary mass

                    * 'r2': if secondary is planet (ignored for stars)

                    * 'age': final system age

                    * 'feh': system metallicity

                    * 'Kwind1': primary K

                    * 'wsat1': primary wsat

                    * 'Kwind2': secondary K

                    * 'wsat2': secondary wsat

                    * 'diff rot coupling timescale1': for primary

                    * 'diff rot coupling timescale2': for secondary

                    * 'Pdisk': the disk lock period for the primary

                    * 'i0': initial obliquity of primary (secondary assumed
                      aligned)

                    * 'e0': initial eccentricity

                    * 'L0': initial secondary angular momentum (all zones)

                    * 'formation age': binary formation age

                    * 'tdisk': disk dissipatino age

                    * 'lgQ1': lgQ corresponding to primary reference phase lag
                      only

                    * 'lgQ2': ditto for secondary

                    * 'satpowerlaw_lag1': refererence phase lag, powerlaw and
                       saturation period for primary phase lag

                    * 'satpowerlaw_lag2': ditto for secondary phase lag

                Any values not in the above list are ignored.

            kwargs:    Passed directly to `InitialConditionSolver.__init__()`

        Returns:
            None
        """

        self._learning_features = learning_features
        super().__init__(**kwargs)
        self._guess_porb_range = MockLearnInitialConditions(ic_model_id,
                                                            reset_ic_model)


    #Meant to be overwritten if needed
    #pylint: disable=no-self-use
    #pylint: disable=unused-argument
    def validate_training_data(self, ml_features, initial_porb):
        """Return True iff the given training data should be used."""

        return True
    #pylint: enable=no-self-use
    #pylint: enable=unused-argument
