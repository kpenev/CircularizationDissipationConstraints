"""Parse the supplementary targets table from Kiefer et. al. 2019."""

import re

import numpy

#Due to copy-pasted data.
#pylint: disable=too-many-statements
def get_supplementary_targets():
    """Return the fully parsed table as a strsuctured array."""

    def parse_names(column_str, destination):
        """Fill destination array with source IDs."""

        index = 0
        id_type = None
        for piece in column_str.split():
            if piece == 'HD':
                id_type = 'HD'
                continue
            elif piece.startswith('BD') or piece.startswith('HD'):
                destination[index] = piece
            else:
                assert id_type == 'HD'
                destination[index] = 'HD' + piece
                id_type = None
            index += 1

    def parse_string_with_errors(column_str, values, plus_errors, minus_errors):
        """
        Parse a column stored as a single string of numbers w/ errors.

        Args:
            values(1-D array):    A pre-allocated array to fill with the parsed
                values.

            plus_errors(1-D array):    A pre-allocated array to fill with the
                parsed errors toward larger values.

            minus_errors(1-D array):    A pre-allocated array to fill with the
                parsed errors toward smaller values.

        Returns:
            None
        """

        val = p_error = m_error = None
        symmetric_error = False
        index = 0
        for piece in column_str.split():
            for sub_piece in piece.split('+-'):
                if val is None:
                    val = float(sub_piece)
                elif sub_piece.startswith('+'):
                    assert p_error is None
                    p_error = float(sub_piece[1:])
                elif sub_piece.startswith('-'):
                    assert m_error is None
                    m_error = float(sub_piece[1:])
                elif sub_piece == '±':
                    assert p_error is None
                    assert m_error is None
                    symmetric_error = True
                elif symmetric_error:
                    p_error = m_error = float(sub_piece)
                    symmetric_error = False
                else:
                    values[index] = val
                    plus_errors[index] = (p_error if p_error is not None
                                          else numpy.nan)
                    minus_errors[index] = (m_error if m_error is not None
                                           else numpy.nan)
                    val = float(sub_piece)
                    p_error = m_error = None
                    index += 1

        values[index] = val
        plus_errors[index] = (p_error if p_error is not None
                              else numpy.nan)
        minus_errors[index] = (m_error if m_error is not None
                               else numpy.nan)

    def parse_string_list(column, values, plus_errors, minus_errors):
        """Parse a column stored as a list of strings (one for each entry)."""

        match_float = '[+-]?[0-9.]+'
        symmetric_errors = re.compile(r'(?P<value>'
                                      +
                                      match_float
                                      +
                                      r')\s*±\s*(?P<error>'
                                      +
                                      match_float
                                      +
                                      r')')
        asymmetric_errors = re.compile(r'(?P<value>'
                                       +
                                       match_float
                                       +
                                       r')\s*[+]\s*(?P<perr>'
                                       +
                                       match_float
                                       +
                                       r')\s*-\s*(?P<merr>'
                                       +
                                       match_float
                                       +
                                       r')')

        for index, value_str in enumerate(column):
            symmetric_match = symmetric_errors.fullmatch(value_str)
            if symmetric_match:
                values[index] = float(symmetric_match['value'])
                plus_errors[index] = float(symmetric_match['error'])
                minus_errors[index] = plus_errors[-1]
                continue
            asymmetric_match = asymmetric_errors.fullmatch(value_str)
            if asymmetric_match:
                values[index] = float(asymmetric_match['value'])
                plus_errors[index] = float(asymmetric_match['perr'])
                minus_errors[index] = float(asymmetric_match['merr'])
                continue
            values[index] = float(value_str)
            plus_errors[index] = numpy.nan
            minus_errors[index] = numpy.nan

    def parse_coordinate(coord_str):
        """Parse the given RA or Dec string."""

        return [tuple(float(v) for v in entry.split(':'))
                for entry in coord_str.split()]

    raw_data = dict(
        #Directly copy-pasted from article to avoid typos
        #pylint: disable=line-too-long
        name=(
            "BD+244697 BD+482155 HD 110833 HD11443 HD 114762 HD 118742 HD 122562 HD 127506 HD 132032 HD 13507 HD 137510 HD 140913 HD 14348 HD 14651 HD 160508 HD 169822 HD 174457 HD 209262 HD 221115 HD 22468 HD 22781 HD 28291 HD 283668 HD 29587 HD 30246 HD 33636 HD 38529 HD 65430 HD 77065 HD 72946 HD 92320 HD 98230"
        ),
        ra=(
            "23:01:39.322 13:50:07.269 12:44:14.545 01:53:04.908 13:12:19.743 13:38:01.953 14:02:21.163 14:30:44.975 14:56:43.930 02:12:54.990 15:25:53.270 15:45:07.449 02:19:52.925 02:22:00.854 17:39:12.696 18:26:10.089 18:50:02.059 22:01:54.121 23:29:09.297 03:36:47.289 03:40:49.524 04:28:37.215 04:27:52.909 04:41:36.318 04:46:30.386 05:11:46.449 05:46:34.913 07:59:33.937 09:00:47.445 08:35:51.266 10:40:56.909 11:18:10.836"
        ),
        dec=(
            "+25:47:16.54 +47:49:15.95 +51:45:33.37 +29:34:43.79 +17:31:01.64 +39:10:41.10 +20:52:52.74 +35:27:13.43 +13:08:57.14 +40:40:06.22 +19:28:50.55 +28:28:11.74 +31:20:14.92 +04:44:48.33 +26:45:27.15 +08:46:39.28 +15:18:41.44 +04:46:13.62 +12:45:37.99 +00:35:15.93 +31:49:34.65 +19:44:26.47 +24:26:41.88 +42:07:06.49 +15:28:19.35 +04:24:12.76 +01:10:05.51 +20:50:38.19 +21:27:13.37 +06:37:21.97 +59:20:33.01 +31:31:44.82"
        ),
        hipparcos_parallax=(
            "20.51 ± 1.33 9.87 ± 1.40 67.20 ± 0.66 51.50 ± 0.23 25.87 ± 0.76 21.74 ± 0.80 18.60 ± 0.72 44.01 ± 0.93 17.86 ± 0.97 37.25 ± 0.55 24.24 ± 0.51 22.27 ± 0.82 17.68 ± 0.45 24.65 ± 0.94 10.83 ± 0.79 34.61 ± 1.39 18.79 ± 0.78 20.12 ± 0.79 18.65 ± 0.78 32.59 ± 0.64 30.51 ± 1.11 21.15 ± 0.77 23.66 ± 1.97 36.27 ± 0.87 21.08 ± 0.86 35.25 ± 1.02 25.46 ± 0.40 42.15 ± 0.71 31.52 ± 1.05 38.11 ± 0.85 23.79 ± 0.78 114.49 ± 0.43"
        ),
        b_minus_v=(
            "1.005 ± 0.036 0.599 ± 0.037 0.936 ± 0.014 0.488 ± 0.009 0.525 ± 0.013 0.698 ± 0.005 0.962 ± 0.010 1.031 ± 0.014 0.636 ± 0.015 0.672 ± 0.007 0.618 ± 0.012 0.612 ± 0.007 0.596 ± 0.015 0.720 ± 0.015 0.543 ± 0.013 0.699 ± 0.005 0.621 ± 0.015 0.687 ± 0.017 0.94 ± 0.00 0.885 ± 0.007 0.845 ± 0.023 0.741 ± 0.014 0.894 ± 0.006 0.633 ± 0.010 0.665 ± 0.006 0.588 ± 0.016 0.773 ± 0.001 0.833 ± 0.008 0.839 ± 0.010 0.710 ± 0.015 0.679 ± 0.015 0.65 ± 0.02"
        ),
        primary_mass=(
            "0.721 ± 0.026 1.068 ± 0.039 0.771 ± 0.011 1.189 ± 0.010 1.147 ± 0.014 0.970 ± 0.005 0.752 ± 0.007 0.703 ± 0.010 1.030 ± 0.015 0.995 ± 0.007 1.048 ± 0.012 1.054 ± 0.007 1.071 ± 0.016 0.950 ± 0.014 1.127 ± 0.014 0.969 ± 0.005 1.045 ± 0.015 0.981 ± 0.016 0.768 ± 0.000 0.810 ± 0.005 0.842 ± 0.019 0.931 ± 0.013 0.803 ± 0.005 1.033 ± 0.010 1.002 ± 0.006 1.079 ± 0.017 0.902 ± 0.001 0.851 ± 0.007 0.846 ± 0.008 0.959 ± 0.014 0.988 ± 0.014 1.016 ± 0.020"
        ),
        #pylint: enable=line-too-long
        orbital_period=(
            '145.081 ± 0.016',
            '90.270 ± 0.019',
            '271.17',
            '1.77',
            '83.9152 ± 0.0028',
            '11.5896 ± 0.0005',
            '2777+100-80',
            '2599',
            '274.33 ± 0.24',
            '4880+210 -190',
            '801.30 ± 0.45',
            '147.968',
            '4740 ± 6',
            '79.4179 ± 0.0021',
            '178.90 ± 0.0074',
            '293.1',
            '840.80 ± 0.05',
            '5430+140 -100',
            '941.03 ± 0.12',
            '1152 ± 44',
            '528.07 ± 0.14',
            '41.66',
            '2558 ± 8',
            '1481 ± 22',
            '990.7 ± 5.6',
            '2128',
            '2136.14 ± 0.29',
            '3138.0',
            '119.1135 ± 0.0027',
            '5814 ± 50',
            '145.402 ± 0.013',
            '3.98'
        ),
        eccentricity=(
            '0.50048 ± 0.00043',
            '0.4375 ± 0.0040',
            '0.784',
            '0.07',
            '0.33 ± 0.15',
            '0.084 ± 0.019',
            '0.71 ± 0.01',
            '0.716',
            '0.0844 ± 0.0024',
            '0.20 ± 0.04',
            '0.3985 ± 0.0073',
            '0.54',
            '0.455 ± 0.004',
            '0.475 ± 0.001',
            '0.5967 ± 0.0009',
            '0.48',
            '0.23 ± 0.01',
            '0.35 ± 0.01',
            '0.517 ± 0.012',
            '0.40 ± 0.22',
            '0.8191 ± 0.0023',
            '0.66',
            '0.577 ± 0.011',
            '0.713 ± 0.006',
            '0.838 ± 0.081',
            '0.48',
            '0.362 ± 0.002',
            '0.32',
            '0.35 ± 0.05',
            '0.495 ± 0.006',
            '0.3226 ± 0.0014',
            '0'
        ),
        min_secondary_mass=(
            '53 ± 3',
            '62.6 ± 0.6',
            '17',
            '71',
            '10.99 ± 0.09',
            '77.8 ± 1.6',
            '24 ± 2',
            '36',
            '70 ± 4',
            '67 ± 9',
            '27.3 ± 1.9',
            '43.2',
            '48.9 ± 1.6',
            '47.0 ± 3.4',
            '48 ± 3',
            '27.2',
            '58.22 ± 0.75',
            '32.3 ± 1.6',
            '89.7 ± 1.4',
            '72 ± 24',
            '13.65 ± 0.97',
            '89',
            '53 ± 4',
            '55.2 ± 9.2',
            '55+20-8',
            '9.3',
            '13.99 ± 0.59',
            '67.8',
            '41 ± 2',
            '60.4 ± 2.2',
            '59.4 ± 4.1',
            '35'
        )
    )

    result = numpy.empty(
        32,
        dtype=(
            [
                ('name', 'S100'),
                ('RA', numpy.float64, (3,)),
                ('Dec', numpy.float64, (3,))
            ]
            +
            [
                (colname, numpy.float64)
                for colname in ['HipPar',
                                'pluserr_HipPar',
                                'minuserr_HipPar',
                                'B-V',
                                'pluserr_B-V',
                                'minuserr_B-V',
                                'Mprimary',
                                'pluserr_Mprimary',
                                'minuserr_Mprimary',
                                'Porb',
                                'pluserr_Porb',
                                'minuserr_Porb',
                                'Ecc',
                                'pluserr_Ecc',
                                'minuserr_Ecc',
                                'M2sini',
                                'pluserr_M2sini',
                                'minuserr_M2sini']
            ]
        )
    )

    parse_names(raw_data['name'], result['name'])
    result['RA'] = parse_coordinate(raw_data['ra'])
    result['Dec'] = parse_coordinate(raw_data['dec'])
    parse_string_with_errors(raw_data['hipparcos_parallax'],
                             result['HipPar'],
                             result['pluserr_HipPar'],
                             result['minuserr_HipPar'])
    parse_string_with_errors(raw_data['b_minus_v'],
                             result['B-V'],
                             result['pluserr_B-V'],
                             result['minuserr_B-V'])
    parse_string_with_errors(raw_data['primary_mass'],
                             result['Mprimary'],
                             result['pluserr_Mprimary'],
                             result['minuserr_Mprimary'])
    parse_string_list(raw_data['orbital_period'],
                      result['Porb'],
                      result['pluserr_Porb'],
                      result['minuserr_Porb'])
    parse_string_list(raw_data['eccentricity'],
                      result['Ecc'],
                      result['pluserr_Ecc'],
                      result['minuserr_Ecc'])
    parse_string_list(raw_data['min_secondary_mass'],
                      result['M2sini'],
                      result['pluserr_M2sini'],
                      result['minuserr_M2sini'])
    return result
#pylint: enable=too-many-statements

if __name__ == '__main__':
    print(get_supplementary_targets())
