#
# Copyright 2021 Lars Pastewka
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import numpy as np

length_units = {'Gm': 1e9, 'Mm': 1e6, 'km': 1000.0, 'm': 1.0, 'mm': 1e-3, 'µm': 1e-6, 'um': 1e-6, 'nm': 1e-9,
                'Å': 1e-10, 'pm': 1e-12, 'fm': 1e-15}
voltage_units = {'GV': 1e9, 'MV': 1e6, 'kV': 1000.0, 'V': 1.0, 'mV': 1e-3, 'µV': 1e-6, 'nV': 1e-9, 'pV': 1e-12,
                 'fV': 1e-15}

units = dict(length=length_units, voltage=voltage_units)


def is_length_unit(s):
    return s in length_units.keys()


def get_unit_conversion_factor(unit1_str, unit2_str):
    """
    Compute factor for conversion from unit1 to unit2.

    Parameters
    ----------
    unit1_str : str
        Name of source unit
    unit2_str : str
        Name of targe unit

    Returns
    -------
    fac : float
        Unit conversion factors. A quantity in unit1 is converted to unit2
        by multiplication with this factor.
    """
    if unit1_str is None:
        raise ValueError('Cannot convert from None unit')
    if unit2_str is None:
        raise ValueError('Cannot convert to None unit')
    if unit1_str == unit2_str:
        return 1
    unit1_kind = None
    unit2_kind = None
    unit_scales = None
    for key, values in units.items():
        if unit1_str in values:
            unit1_kind = key
            unit_scales = values
        if unit2_str in values:
            unit2_kind = key
            unit_scales = values
    if unit1_kind is None:
        raise ValueError(f"Unknown unit '{unit1_str}'.")
    if unit2_kind is None:
        raise ValueError(f"Unknown unit '{unit2_str}'.")
    if unit1_kind != unit2_kind:
        raise ValueError(f"Unit '{unit1_str}' is of kind {unit1_kind} while unit '{unit2_str}' is of kind {unit2_str}."
                         "I cannot convert between the two.")
    return unit_scales[unit1_str] / unit_scales[unit2_str]


def mangle_length_unit_utf8(unit):
    """
    Convert unit string to normalized UTF-8 unit string, e.g. converts 'um'
    to 'µm' and makes sure 'µ' is MICRO SIGN (00B5) and not GREEK SMALL LETTER
    MU (03BC).

    Parameters
    ----------
    unit : str
        Name of unit

    Returns
    -------
    output_unit : str
        Mangled name of unit
    """
    if isinstance(unit, str):
        unit = unit.strip()
    else:
        unit = unit.decode('utf-8').strip()
    if unit == '':
        return None
    elif unit == 'A':
        return 'Å'
    elif unit == 'μm' or unit == 'um' or unit == '~m':
        return 'µm'
    else:
        return unit


def mangle_length_unit_ascii(unit):
    """
    Convert unit string to ASCII representation, e.g. converts 'µm'
    to 'um'.

    Parameters
    ----------
    unit : str
        Name of unit

    Returns
    -------
    output_unit : str
        Mangled name of unit
    """
    unit = unit.strip()
    if unit == '':
        return None
    elif unit == 'Å':
        return 'A'
    elif unit == 'μm' or unit == 'µm' or unit == '~m':
        return 'um'
    else:
        return unit


def suggest_length_unit(scale, lower_in_meters, upper_in_meters):
    """
    Suggest a length unit for representing data in a certain range.
    E.g. data in the range from 1e-3 to 1e-2 m is best represented by um.

    Parameters
    ----------
    scale : str
        'linear': displaying data on a linear axis
        'log' displaying data on a log-space axis
    lower_in_meters : float
        Lower bound of range in meters
    upper_in_meters : float
        Upper bound of range in meters

    Returns
    -------
    unit : str
        Suggestion for the length unit
    """
    if scale == 'linear':
        v = max(abs(lower_in_meters), abs(upper_in_meters))
        m10 = 3 * int(np.floor(np.log10(v) / 3))
    elif scale == 'log':
        u10 = int(np.ceil(np.log10(upper_in_meters)))
        l10 = int(np.floor(np.log10(lower_in_meters)))
        m10 = 3 * int(np.ceil((l10 + u10) / 6) - 1)
    else:
        raise ValueError(f"Unknown scale parameter '{scale}'.")

    fac = 10 ** m10
    minfac = 1
    minunit = 'm'
    maxfac = 1
    maxunit = 'm'
    for key, value in length_units.items():
        if value < minfac:
            minfac = value
            minunit = key
        if value > maxfac:
            maxfac = value
            maxunit = key
        if value == fac:
            return key

    # We could not identify any unit from our list
    if fac < minfac:
        return minunit
    elif fac > maxfac:
        return maxunit
    else:
        raise RuntimeError(f'Cannot find unit for scale prefix {fac}.')


def suggest_length_unit_for_data(scale, data, unit):
    """
    Suggest a length unit for representing a data set.

    Parameters
    ----------
    scale : str
        'linear': displaying data on a linear axis
        'log' displaying data on a log-space axis
    data : array_like
        Data set that needs representing (e.g. in a color bar)
    unit : str
        Unit of the data set

    Returns
    -------
    unit : str
        Suggestion for the length unit
    """
    mn, mx = np.nanmin(data), np.nanmax(data)
    if np.isnan(mn) or np.isnan(mx):
        raise ValueError('Data does not contain values that are not NaN.')
    if not (np.isfinite(mn) and np.isfinite(mx)):
        # There are Inf or -Inf values in the data, we just return the original unit.
        return unit
    fac = get_unit_conversion_factor(unit, 'm')
    return suggest_length_unit(scale, fac * mn, fac * mx)
