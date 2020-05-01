"""Define the interface to working with all WEBDA data for a cluster."""

import os.path
from glob import glob

import pandas
from astropy import units

from webda.read_table import read_table

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

    def __init__(self, cluster_name):
        """Read and organize the data for the given cluster for use."""

        self.names = [cluster_name]
        webda_info = self._cluster_info.loc[cluster_name]
        self.age = 10.0**webda_info['Age'] * units.yr
        self.feh = webda_info['Fe/H']
        self._star_data = self._read_data(
            os.path.join(
                self._webda_dir,
                cluster_name.lower().replace(' ', '_')
            )
        )

    def get_color_magnitude(self, binaries='exclude'):
        """
        Return the color/magnitude measurements to use for mass fitting.

        Args:
            binaries(str):    Determines the set of stars for which data is
                returned {'include', 'exclude', 'only'}

                  * `'include'`: result includes all stars not flagges as
                    non-members (binaries and singles).

                  * `'exclude'`: result excludes all flagged non-mebers and
                    stars flagged as binaries.

                  * `'only'`: result includes only stars known to be binaries.

        Returns:
            pandas.DataFrame:
                Photometric data for the selected collection of stars, one
                entry per star. For multiple published values, the average,
                weighted with the given `N` is used.
        """

if __name__ == '__main__':
    hyades = Cluster('Melotte 25')
    print('Age: ' + repr(hyades.age.to_value('Gyr')))
    print('[Fe/H]: ' + repr(hyades.feh))
