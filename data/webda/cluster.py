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

    _tables_with_errors = ['melotte_25/elem.orb']
    _drop_stars = {'Melotte 25': ['0169']}

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

        def combine_to_drop(current_to_drop, extra_to_drop):
            """
            Combine two lists of indices to drop.

            Args:
                current_to_drop(None or pandas index):    The current list of
                    stars marked for exclusion from the result, or None if no
                    stars have been identified so far.

                extra_to_drop(pandas index):    The part of the index of a WEBDA
                    table identifying stars to drop. Could be either just star
                    numbers or star numbers and references, in which case the
                    references are ignored.

            Returns:
                pandas index:
                    The star numbers contained in either input index.
            """

            new_stars = (
                extra_to_drop.get_level_values(0).drop_duplicates()
            )
            if current_to_drop is None:
                return new_stars
            return current_to_drop.union(new_stars)

        def get_raw_data():
            """
            Return all color information for the stars to include in result.

            May include multiple entries per star if multiple measurements
            exist.
            """

            to_drop = None
            if 'NM' in self._star_data:
                to_drop = self._star_data['NM'].index
            if 'prob.mu' in self._star_data:
                to_drop = combine_to_drop(
                    to_drop,
                    self._star_data['prob.mu'].index[
                        self._star_data['prob.mu']['Prob']
                        <
                        membership_threshold
                    ]
                )
            if binaries == 'exclude' and 'SB' in self._star_data:
                to_drop = combine_to_drop(
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

if __name__ == '__main__':
    import warnings
    warnings.simplefilter('error')

    hyades = Cluster('Melotte 25')
    print('Age: ' + repr(hyades.age.to_value('Gyr')))
    print('[Fe/H]: ' + repr(hyades.feh))
    print(hyades.get_color_magnitude())
