"""Define the interface to working with all WEBDA data for a cluster."""

import os.path
from glob import glob

import pandas
import numpy
from astropy import units

#relevant path should be added to sys.path by caller
#pylint: disable=import-error
from webda.read_table import read_table
#pylint: enable=import-error

class Cluster:
    """
    Convenient interface to the relevant WEBDA data for the stars in a cluster.

    Attributes:
        names([str]):    The various names the cluster goes by.

        age(astropy Quantity):    The age of the cluster per WEBDA.

        feh(float):    The [Fe/H] of the cluster per WEBDA.

        _cluster_info(pandas.DataFrame):    The parsed information about
            clusters from WEBDA.

        _tables_with_errors([]):    A list of the tables which alternate values
            lines with error lines.

        _drop_stars(dict):    List of stars to drop by cluster.

        _star_data(dict):    The parsed WEBDA data for the cluster organized in
            a dictionary with keys the filenames and the values pandas.DataFrame
            containing the data from the corresponding file.

        star_data
    """

    _webda_dir = os.path.dirname(__file__)
    _cluster_info = pandas.read_html(
        os.path.join(_webda_dir, 'cluster_table.html'),
        index_col='Cluster_name'
    )[0]

    _tables_with_errors = ['elem.orb']
    _drop_stars = {'Melotte 25': ['0169']}

    @staticmethod
    def _unique_combine_star_lists(index1, index2):
        """
        Combine two pandas indices excl. duplicates, handling special cases

        Args:
            index1(None or pandas index):    The first list of indices, or
                None if no stars have been identified so far.

            inedx2(pandas index):    The part of the index of a WEBDA
                table to join with the first. Could be either just star
                numbers or star numbers and references, in which case the
                references are ignored.

        Returns:
            pandas index:
                The star numbers contained in either input index.
        """

        new_stars = (
            index2.get_level_values(0).drop_duplicates()
        )
        if index1 is None:
            return new_stars
        return index1.union(new_stars)

    def _read_data(self, data_dir):
        """Read the cluster data into pandas.DataFrame objects."""

        def is_table(fname):
            """Return True iff the given filename is a WEBDA table."""

            return not os.path.splitext(fname)[1] in ['.rst', '.tar']

        result = dict()
        for fname in filter(is_table, glob(os.path.join(data_dir, '*'))):
            table = os.path.basename(fname)
            result[table] = read_table(
                fname,
                has_errors=(table in self._tables_with_errors),
                drop_stars=self._drop_stars[self.names[0]]
            )
        return result

    @staticmethod
    def _select_orbital_elements(orbit):
        """
        Select which of the two orbits is "better".

        Args:
            orbit(pandas.DataFrame):    The values and errors of the orbital
                elements from all references for a single star.

        Returns:
            pandas.DataFrame:
                The selected entry from the input.
        """

        print('Selectnig best orbit among: ')
        print(orbit)
        porb_best = orbit['err_Po'].idxmin()
        k1_best = orbit['err_K1'].idxmin()

        print('Po best: ' + repr(porb_best))
        print('K1 best: ' + repr(k1_best))
        if not isinstance(porb_best, float):
            print('best err_Po: ' + repr(orbit['err_Po'][porb_best]))
        if not isinstance(k1_best, float):
            print('best err_K1: ' + repr(orbit['err_K1'][k1_best]))


        if isinstance(porb_best, float):
            assert numpy.isnan(porb_best)
            best = k1_best
        else:
            if isinstance(k1_best, float):
                assert numpy.isnan(k1_best)
                best = porb_best
            else:
                if (
                        numpy.isnan(orbit['err_Po'][porb_best])
                        or
                        numpy.isnan(orbit['err_K1'][porb_best])
                ):
                    best = k1_best
                elif (
                        numpy.isnan(orbit['err_K1'][k1_best])
                        or
                        numpy.isnan(orbit['err_Po'][k1_best])
                ):
                    best = porb_best
                elif k1_best == porb_best:
                    best = porb_best
                else:
                    if (
                            orbit['err_K1'][porb_best]/orbit['err_K1'][k1_best]
                            >
                            orbit['err_Po'][k1_best]/orbit['err_Po'][porb_best]
                    ):
                        best = k1_best
                    else:
                        best = porb_best

        result = orbit.loc[[best]].reset_index(level=0, drop=True)
        return result

    def __init__(self, cluster_name):
        """Read and organize the data for the given cluster for use."""

        self.names = [cluster_name]
        webda_info = self._cluster_info.loc[cluster_name]
        #False positive
        #pylint: disable=no-member
        self.age = 10.0**webda_info['Age'] * units.yr
        #pylint: enable=no-member
        self.feh = webda_info['Fe/H']
        self._star_data = self._read_data(
            os.path.join(
                self._webda_dir,
                cluster_name.lower().replace(' ', '_')
            )
        )

    def get_nonmembers(self, membership_threshold=0.5):
        """Return a list of the IDs of all non-stars present in cluster data."""

        if 'NM' in self._star_data:
            non_members = self._star_data['NM'].index
        if 'prob.mu' in self._star_data:
            non_members = self._unique_combine_star_lists(
                non_members,
                self._star_data['prob.mu'].index[
                    self._star_data['prob.mu']['Prob']
                    <
                    membership_threshold
                ]
            )
        return non_members


    def get_color_magnitude(self,
                            binaries='exclude',
                            membership_threshold=0.5,
                            undefined_nobs=1):
        """
        Return the color/magnitude measurements to use for mass fitting.

        Args:
            binaries(str):    Determines the set of stars for which data is
                returned {'include', 'exclude', 'only'}

                  * `'include'`: result includes all stars not flagges as
                    non-members (binaries and singles).

                  * `'exclude'`: result excludes all flagged non-mebers and
                    stars flagged as binaries.

                  * `'only'`: result includes only stars known to be binaries,
                    and cluster members.

            membership_threshold(float):    The minimum membership probability
                to consider the star a member.

            undefined_nobs(float):    The weight to give to photometric
                rows with unknown number of measurements.

        Returns:
            pandas.DataFrame:
                Photometric data for the selected collection of stars, one
                entry per star. For multiple published values, the average,
                weighted with the given `N` is used.
        """

        def get_raw_data():
            """
            Return all color information for the stars to include in result.

            May include multiple entries per star if multiple measurements
            exist.
            """

            to_drop = self.get_nonmembers(membership_threshold)
            if binaries == 'exclude' and 'SB' in self._star_data:
                to_drop = self._unique_combine_star_lists(
                    to_drop,
                    self._star_data['SB'].index
                )

            result = self._star_data['ubv.peo'].drop(index=to_drop,
                                                     level=0)
            if binaries == 'only':
                result = result.loc[self._star_data['SB'].index]

            return result.copy()

        def combine_photometry(star_data_frame):
            """Calculate weighted mean photometry for a star."""

            result_data = dict(N=star_data_frame['N'].sum())
            for column in star_data_frame.keys():
                if column != 'N':
                    finite = numpy.isfinite(star_data_frame[column])
                    if finite.any():
                        result_data[column] = numpy.average(
                            star_data_frame[column][finite],
                            weights=star_data_frame['N'][finite]
                        )
                    else:
                        result_data[column] = numpy.nan
            return pandas.DataFrame(
                data=result_data,
                index=[0]
            )

        assert binaries in ['include', 'exclude', 'only']
        data = get_raw_data()
        data.loc[data['N'] == 0, 'N'] = undefined_nobs
        result = data.groupby(level=0).apply(combine_photometry)
        result.index = result.index.droplevel(1)
        return result

    def get_binary_orbits(self, membership_threshold=0.5):
        """Return the orbital elements of all member binaries."""

        return self._star_data[
            'elem.orb'
        ].groupby(
            level=0
        ).apply(
            self._select_orbital_elements
        ).drop(
            index=self.get_nonmembers(membership_threshold),
            level=0
        )

    def get_binaries(self,
                     membership_threshold=0.5,
                     require_orbit=False,
                     require_photometry=False,
                     **color_magnitude_kwargs):
        """
        Return available information for binaries.

        Args:
            select_orbital_elements(callable):    Should be a callable which
                takes all available orbital elements entries for a system and
                returns the best. This is passed directly to groupby().apply of
                the pandas DataFrame of orbital elements.

            membership_threshold(float):    The minimum membership probability
                which will still count the star as a member.

            require_orbit(bool):    Should the result include only binaries for
                which orbital elements are available?

            require_photometry(bool):    Should the result include only binaries
                for which photometry is available?

            color_magnitude_kwargs:    Any arguments to pass directly to
                self.get_color_magnitude(), when collecting photometry to
                include for the binaries.

        Returns:
            pandas.DataFrame:
                Indexed by star number and columns containing a selected "best"
                values for all relevant binary information.
        """

        if require_orbit:
            if require_photometry:
                merge_type = 'inner'
            else:
                merge_type = 'left'
        else:
            if require_photometry:
                merge_type = 'right'
            else:
                merge_type = 'outer'

        return pandas.merge(
            self.get_binary_orbits(
                membership_threshold
            ).reset_index(
                level='Ref'
            ),
            self.get_color_magnitude(
                binaries='only',
                membership_threshold=membership_threshold,
                **color_magnitude_kwargs
            ),
            how=merge_type,
            left_index=True,
            right_index=True,
            suffixes=['orb', 'cm']
        )

def examples():
    """Run examples without polluting global scope."""

    hyades = Cluster('Melotte 25')

    binaries = hyades.get_binaries(require_orbit=True)
    print(binaries)


    cmd_data_singles = hyades.get_color_magnitude()
    cmd_data_binaries = hyades.get_color_magnitude(binaries='only')
    print('Age: ' + repr(hyades.age.to_value('Gyr')))
    print('[Fe/H]: ' + repr(hyades.feh))

    pyplot.plot(cmd_data_singles['BV'], -cmd_data_singles['V'], 'ok')
    pyplot.plot(cmd_data_binaries['BV'], -cmd_data_binaries['V'], 'xr')
    pyplot.plot(binaries['BV'], -binaries['V'], '+g')
    pyplot.show()

if __name__ == '__main__':
    from matplotlib import pyplot
#    import warnings
#    warnings.simplefilter('error')
    examples()
