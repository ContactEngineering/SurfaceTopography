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


height_units = {'m': 1.0, 'mm': 1e-3, 'um': 1e-6, 'µm': 1e-6, 'nm': 1e-9, 'Å': 1e-10}
voltage_units = {'kV': 1000.0, 'V': 1.0, 'mV': 1e-3, 'µV': 1e-6, 'nV': 1e-9}

units = dict(height=height_units, voltage=voltage_units)


###


def get_unit_conversion_factor(unit1_str, unit2_str):
    """
    Compute factor for conversion from unit1 to unit2.
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
    unit = unit.strip()
    if unit == '':
        return None
    elif unit == 'Å':
        return 'A'
    elif unit == 'μm' or unit == 'µm' or unit == '~m':
        return 'um'
    else:
        return unit
