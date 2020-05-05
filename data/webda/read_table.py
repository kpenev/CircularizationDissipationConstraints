"""Functions for reading a single table from a cluster dataset."""

import warnings

import pandas
import numpy

def read_table(fname,
               has_errors=False,
               drop_stars=None,
               force_unique_index=False):
    """
    Return a pandas DataFrame for a file of quantities with no errors.

    Args:
        fname(str):    The filename to read from. Should include the cluster
            name, or be an absolute path.

        has_errors(bool):    Whether the file being read contains alternating
            rows of values and error estimates.

        drop_stars(None or [str]):    A list of stellar IDs to exclude from the
            parsed tabled.

        force_unique_index(bool):    If True, repeating (star, reference)
            combinations are dropped from the result, keeping only the last
            entry for each. This ensures that the resulting DataFrame has a
            unique index.

    Returns:
        pandas.DataFrame:
            The data in the file indexed by star number (and by
            reference number if multiref). For files with error estimates, the
            data frame has twice as many columns as the file, with the second
            set of columns prefixed by `'err_'` containing the error estimates.
    """


    with open(fname, 'r') as table_file:
        colnames = table_file.readline().split(None, 2)
        has_references = len(colnames) > 1 and colnames[1] == 'Ref'
    data = pandas.read_csv(
        fname,
        sep='\t',
        header=0,
        skiprows=[1],
        dtype={'No': str, 'Ref': int, 'Vo': float},
        index_col=False,
        skip_blank_lines=False,
        encoding='ascii',
        low_memory=False,
        na_values=['(fixed)', 'fixed', '.', 'var']
    )
    if has_references:
        data = data.set_index(['No', 'Ref'])
    else:
        data = data.set_index('No')
    if drop_stars:
        if has_references:
            data = data.drop(index=drop_stars, level=0, errors='ignore')
        else:
            data = data.drop(index=drop_stars, errors='ignore')
    if has_errors:
        value_mask = numpy.logical_not(data.index.duplicated(keep='first'))
        error_mask = numpy.logical_not(data.index.duplicated(keep='last'))

        errors = data[error_mask]

        in_use = numpy.logical_or(value_mask, error_mask)
        if not in_use.all():
            raise KeyError(
                'Multiple entries in %s for ' % repr(fname)
                +
                repr(data.index[numpy.logical_not(in_use)].drop_duplicates())
            )

        no_error = numpy.logical_and(value_mask, error_mask)
        if no_error.any():
            warnings.warn(
                'No error estimate in %s for ' % repr(fname)
                +
                repr(data.index[no_error].drop_duplicates())
            )
            errors = errors.copy()
            errors[no_error[error_mask]] = None

        return pandas.concat(
            [
                data[value_mask],
                errors.rename(columns=lambda colname: 'err_' + colname)
            ],
            axis=1
        )

    if force_unique_index and not data.index.is_unique:
        to_drop = data.index.duplicated(keep='last')
        warnings.warn(
            'Non-unique index in %s, dropping:\n' % repr(fname)
            +
            str(data[to_drop])
        )
        data = data[numpy.logical_not(to_drop)]

    return data

if __name__ == '__main__':
    spectroscopic_binaries = read_table('webda/melotte_25/SB', has_errors=False)
    adel_coo = read_table('webda/melotte_25/adel.coo', has_errors=True)
    orb_elem = read_table('webda/melotte_25/elem.orb',
                          has_errors=True,
                          force_unique_index=True,
                          drop_stars=['0169'])
    print(orb_elem.dtypes)
    print(80*'=')
    print(spectroscopic_binaries)
    print(adel_coo)
    print(orb_elem)
    print(orb_elem.loc['0141'])
    print(orb_elem.loc['0072'])
    try:
        print(orb_elem.loc['0169'])
    except KeyError:
        assert True
