#!/usr/bin/env python3

import os.path
from csv import QUOTE_NONE

from pandas import read_csv, merge

class MultipleStarCatalogue:
    """Interface to the multiple star catalogue."""

    def __init__(self):
        """Parse the catalogue files."""

        def read_file(fname, names_row):
            """Read the given file assuming column names at the given line."""

            dtype=dict()
            if fname in ['subsystems', 'orbits']:
                dtype['Level'] = int
            elif fname == 'index':
                dtype=str
            return read_csv(
                os.path.join(data_dir, fname + '.tsv'),
                header=0,
                sep='|',
                skiprows=lambda row: (row < names_row
                                      or
                                      row in [names_row + 1, names_row + 2]),
                quoting=QUOTE_NONE,
                skipinitialspace=True,
                dtype=dtype
            )

        data_dir = os.path.dirname(__file__)
        tables = {
            fname: read_file(fname, skiprow)
            for fname, skiprow in [('index', 42),
                                   ('orbits', 58),
                                   ('systems', 52),
                                   ('subsystems', 60),
                                   ('remarks', 30)]
        }

        for tbl_name, tbl in tables.items():
            print(tbl_name + ':')
            print(tbl)

        self.index = tables['index']
        print(self.index['HD'].array)

        self.data = merge(tables['subsystems'],
                          tables['orbits'],
                          how='outer',
                          on=('IDS', 'Level'))

        print('Data:')
        print(self.data)

    def __call__(self, **identifier):
        """
        Retrun all available data for a star with a given identifier.

        Args:
            identifier:    Should be one of the columns from the index table
        """

        assert len(identifier) == 1
        id_type, id_value = next(iter(identifier.items()))
        ids = self.index[self.index[id_type] == id_value]['IDS']
        print('Index result: ' + repr(ids))
        if ids.empty:
            return None
        else:
            return self.data[self.data['IDS'] == ids.array[0]]

if __name__ == '__main__':
    msc = MultipleStarCatalogue()
    result = msc(HD='27691')
    result = msc(HD='27749')
    print(result)
    print('Masses: ' + repr((result['Mass1'], result['Mass2'])))
