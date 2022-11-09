import numpy
from manual_exoplanet_data import data as manual_data
from astropy.units import Unit, Quantity

class Structure:
    """An empty class used only to hold user defined attributes."""

    def __init__(self, **initial_attributes):
        """Create a class with (optionally) initial attributes."""

        for attribute_name, attribute_value in initial_attributes.items():
            setattr(self, attribute_name, attribute_value)

    def format(self, prefix=''):
        """Generate a tree-like representation of self."""

        result = ''
        for attr_name in dir(self):
            if attr_name[0] != '_':
                attribute = getattr(self, attr_name)
                if isinstance(attribute, Structure):
                    result += (prefix
                               +
                               '|-'
                               +
                               attr_name
                               +
                               '\n'
                               +
                               attribute.format(prefix + '| '))
                else:
                    result += (prefix
                               +
                               '|-'
                               +
                               attr_name
                               +
                               ': '
                               +
                               str(attribute)
                               +
                               '\n')
        return result
def convert_nasa_unit_to_astropy(unit_str):
    """Return the astropy unit matching the one specified in input file."""

    print('Unit str: ' + repr(unit_str))
    if unit_str in ['days', 'hrs']:
        unit_str = unit_str[:-1]
    elif unit_str == 'decimal degrees':
        unit_str = 'degree'
    elif (
            unit_str in ['dex', 'Earth flux', 'sexagesimal']
            or
            unit_str.startswith('log10(')
            or
            unit_str.startswith('log(')
    ):
        return None
    elif unit_str == 'Solar mass':
        unit_str = 'solMass'
    elif unit_str == 'Solar radii':
        unit_str = 'solRad'
    elif unit_str.endswith(' mass'):
        unit_str = unit_str.split()[0].lower() + 'Mass'
    elif unit_str.endswith(' radii'):
        unit_str = unit_str.split()[0].lower() + 'Rad'
    elif unit_str.startswith('percent'):
        return 0.01

    print('Converted to: ' + repr(unit_str))

    return Unit(unit_str)

def read_ages(nasa_planets,
              age_file_standard='inputs/versioned/getages.txt',
              age_file_manual_density='inputs/versioned/getages_nodensity.txt',
              manual_densities='inputs/versioned/age_variables_nodensity.txt'):
    """
    Complete the NASA exoplanet archive planets with age information.

    Args:
        - nasa_planets: The planets read from a CSV file downloaded from the
                        NASA exoplanet archive. On output, this gets updated
                        with the information from the various input files.
        - age_file_standard: The name of the file containing the derived
                             ages.
        - age_file_manual_density: The name of the file with ages derived
                                   from manually extracted densities.
        - manual_densities: The name of the file containing the manually
                            extracted densities themselves.

    Returns: None
    """

    def read_file(filename, columns):
        """
        Read one of the input files and update nasa_planets.

        Args:
            - filename: The name of the file to read.
            - columns: a dictionary of the quantities to read from the file
                       (keys) and the columns that contain them. The quantity
                       pl_hostname must be among the columns.

        Returns: None
        """

        hostname_list = list(nasa_planets.pl_hostname)
        num_systems = len(hostname_list)
        for quantity, column in columns.items():
            if not hasattr(nasa_planets, quantity):
                setattr(nasa_planets,
                        quantity,
                        numpy.full((num_systems,), numpy.nan))
        with open(filename, 'r') as input_file:
            for line in input_file:
                entries = line.split()
                host = entries[columns['pl_hostname']]
                system_index = 0
                while (
                        system_index < len(hostname_list)
                        and
                        (
                                not hostname_list[system_index].startswith(host)
                                or
                                (
                                        len(hostname_list[system_index]) > len(host)
                                        and
                                        hostname_list[system_index][len(host)] != ' '
                                )
                        )
                ):
                    system_index += 1
                if system_index == len(hostname_list):
                    continue
                for quantity, column in columns.items():
                    if quantity != 'pl_hostname':
                        try:
                            entry_val = int(entries[column])
                        except ValueError:
                            entry_val = float(entries[column])
                        getattr(nasa_planets, quantity)[system_index] = (
                            entry_val
                        )

    age_file_columns = dict(pl_hostname=0,
                            st_mass=2,
                            st_masserr1=3,
                            st_rad=6,
                            st_raderr1=7,
                            st_age=10,
                            st_ageerr1=11,
                            st_lum=14,
                            st_lumerr1=15)

    density_file_columns = dict(pl_hostname=0,
                                st_dens=10,
                                st_denserr1=11,
                                st_denserr2=12)

    read_file(age_file_standard, age_file_columns)
    read_file(age_file_manual_density, age_file_columns)
    read_file(manual_densities, density_file_columns)

def read_nasa_planets(csv_filename,
                      eliminate=('SWEEPS-11',
                                 'HD 41004 B',
                                 'PSR J1719-1438',
                                 'K2-22',
                                 'WASP-19 B',
                                 'HATS-18'),
                      fill_missing=manual_data,
                      need_ages=True,
                      add_units=False):
    """
    Read a CSV file downloaded from the NASA Exoplanet Archive to a dict.

    Args:
        csv_filename:    The name of the comma separated file downloaded from
            http://exoplanetarchive.ipac.caltech.edu.

    Returns:
        A structure with the column names as attributes containing the
        corresponding values properly formatted.
    """

    def do_eliminate():
        """Eliminate the systems listed in eliminate."""
        system_names = []
        for i in range(1, len(data)):
            name = data[i][0]
            if type(name) is numpy.bytes_:
                system_names = system_names + [name.decode()]
        # system_names = [name.decode() for name in data[:, 1]]
        # The above for loop is written instead of the code system_names = [name.decode() for name in data[:, 1]]
        # couple of name in data[:, 1] were not decoded by decode() method, since they were not numpy.bytes_
        # object. So, I have included a checking command: if type(name) is numpy.bytes_

        delete_indices = []
        for system in eliminate:
            if system in system_names:
                delete_indices.append(system_names.index(system))
        a = numpy.delete(data, delete_indices, 0)
        return a

    def do_fill_missing(result):
        """Add the data from fill_missing to result."""

        system_names = list(
            getattr(
                result,
                (
                    'hostname' if hasattr(result, 'hostname')
                    else 'fpl_hostname'
                )
            )
        )
        for fill_system in fill_missing:
            try:
                fill_index = system_names.index(fill_system['pl_hostname'])
            except ValueError:
                continue
            for quantity, value in fill_system.items():
                if hasattr(result, quantity):
                    target_column = getattr(result, quantity)
                    if isinstance(target_column, Quantity):
                        unit = target_column.unit
                    else:
                        unit = 1
                    target_column[fill_index] = (value * unit)




    with open(csv_filename, 'r') as csv_file:
        while csv_file.readline()[0] == '#':
            data_start = csv_file.tell()
        csv_file.seek(data_start)
        data = numpy.genfromtxt(csv_file,
                                delimiter=',',
                                dtype=None,
                                comments=None)
    data_columns = []
    for i in data[0]:
        if type(i) is numpy.bytes_:
            data_columns = data_columns + [i.decode()]

    # data_columns = [col.decode() for col in data[0]]
    # the above loop was written instead of data_columns = [col.decode() for col in data[0]]

    if eliminate:
        data = do_eliminate()

    result = Structure()
    string_columns = ['pl_hostname',
                      'hostname',
                      'pl_name',
                      'pl_discmethod',
                      'discoverymethod',
                      'disc_facility',
                      'soltype',
                      'pl_refname',
                      'st_refname',
                      'sy_refname',
                      'rastr',
                      'ra',
                      'rowupdate',
                      'pl_pubdate',
                      'releasedate',
                      'decstr',
                      'pl_bmassprov',
                      'st_optband',
                      'rowupdate',
                      'pl_letter',
                      'pl_tsystemref',
                      'pl_locale',
                      'pl_facility',
                      'pl_telescope',
                      'pl_instrument',
                      'pl_publ_date',
                      'hd_name',
                      'hip_name',
                      'st_spstr',
                      'st_metratio',
                      'st_optmagband',
                      'st_nirmagband',
                      'st_spt',
                      'swasp_id']
    column_name_list = []
    with open(csv_filename, 'r') as csv_file:
        for line in csv_file:
            if line[0] != '#':
                continue
            entries = line.strip().rstrip(')').split()
            if len(entries) < 4 or entries[1] != 'COLUMN':
                continue
            column_name = entries[2].strip(':')
            column_index = data_columns.index(column_name)
            column_values = data[:, column_index][1:]
            if add_units:
                if entries[-1][-1] == ']':
                    column_units = convert_nasa_unit_to_astropy(
                        line.strip().rstrip(')').rsplit('[', 1)[-1][:-1]
                    )
                else:
                    column_units = None
            if (
                    column_name in string_columns
                    or
                    column_name[0] == 'f' and column_name[1:] in string_columns
                    or
                    column_name.endswith('_str')
                    or
                    column_name.endswith('link')
            ):
                column_values = [v.decode() for v in column_values]
            else:
                print('column_name: ' + repr(column_name))
                column_values = [
                    numpy.nan if v == b'' else float(v)
                    for v in data[:, column_index][1:]
                ]
            column_values = numpy.array(column_values)

            if add_units and column_units is not None:
                print(column_name + ' units: ' + repr(column_units))
                column_values *= column_units
            setattr(
                result,
                column_name,
                column_values
            )
            column_name_list.append(column_name)

    if fill_missing:
        do_fill_missing(result)

    if need_ages:
        read_ages(result)

    for column_name in column_name_list:
        if column_name.endswith('err2'):
            column = getattr(result, column_name)
            nan_indices = numpy.isnan(column)
            column[nan_indices] = -getattr(result,
                                           column_name[:-1] + '1')[nan_indices]

    return result
