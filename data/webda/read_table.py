"""Functions for reading a single table from a cluster dataset."""

import os.path
import warnings

import pandas
import numpy

_data_dir = os.path.dirname(__file__)

def read_file(fname, multiref=True, has_errors=False, drop_stars=None):
    """
    Return a pandas DataFrame for a file of quantities with no errors.

    Args:
        fname(str):    The filename to read from. Should include the cluster
            name, or be an absolute path.

        multiref(bool):    Whether the file has a single entry per pars (False)
            or many entries per star coming from multiple references. Note that
            if (star, reference number) is duplicated, only the last instance is
            kept.

        has_errors(bool):    Whether the file being read contains alternating
            rows of values and error estimates.

    Returns:
        pandas.DataFrame or (pandas.DataFrame, pandas.DataFrame):
            The data in the file indexed by star number (and by
            reference number if multiref). Two DataFrame objects are returned
            for files with error estimates, the first containing the values and
            the second cantaining the errors.
    """

    data = pandas.read_csv(
        os.path.join(_data_dir, fname),
        sep='\t',
        header=0,
        skiprows=[1],
        index_col=((0, 1) if multiref else [0]),
        skip_blank_lines=False,
        encoding='ascii',
        low_memory=False
    )
    if drop_stars:
        data = data.drop(index=drop_stars, level=0)
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

        return data[value_mask], errors

    if not data.index.is_unique:
        to_drop = data.index.duplicated(keep='last')
        warnings.warn(
            'Non-unique index in %s, dropping:\n' % repr(fname)
            +
            str(data[to_drop])
        )
        data = data[numpy.logical_not(to_drop)]

    return data

if __name__ == '__main__':
    print(read_file('hyades/SB', False))
    print(read_file('hyades/adel.coo', True))
    orb_elem, orb_elem_err = read_file('hyades/elem.orb', True, True, drop_stars=['0169'])
    print(orb_elem)
    print(orb_elem_err)
    print(orb_elem.loc['0141'])
    print(orb_elem_err.loc['0141'])
    print(orb_elem.loc['0095'])
    print(orb_elem_err.loc['0095'])
    try:
        print(orb_elem.loc['0169'])
    except KeyError:
        assert(True)
    try:
        print(orb_elem_err.loc['0169'])
        assert False
    except KeyError:
        assert(True)
