#
# Copyright 2020-2022 Lars Pastewka
#           2019 Antoine Sanner
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
import pytest
from NuMPI.Testing.Fixtures import serial_only

from SurfaceTopography.Support.UnitConversion import (
    get_unit_conversion_factor,
    is_length_unit,
    mangle_length_unit_ascii,
    mangle_length_unit_utf8,
    suggest_length_unit,
    suggest_length_unit_for_data,
)

pytestmark = serial_only


def test_suggest_length_unit():
    assert suggest_length_unit("log", 1e-3, 1e-2) == "mm"
    assert suggest_length_unit("log", 1e-6, 1e-5) == "µm"

    assert suggest_length_unit("linear", 0, 1e-2) == "mm"

    assert suggest_length_unit("log", 1e-9, 1) == "µm"
    assert suggest_length_unit("log", 1e-9, 10) == "µm"
    assert suggest_length_unit("log", 1e-9, 100) == "µm"
    assert suggest_length_unit("log", 1e-9, 100) == "µm"
    assert suggest_length_unit("log", 1e-9, 10000) == "mm"
    assert suggest_length_unit("log", 1e-9, 100000) == "mm"


def test_unit_outside_range():
    assert suggest_length_unit("log", 1e-21, 1e-18) == "fm"
    assert suggest_length_unit("log", 1e12, 1e15) == "Gm"


def test_nan_and_inf():
    assert suggest_length_unit_for_data("log", [1e-9, 10], "m") == "µm"
    assert suggest_length_unit_for_data("log", [1e-9, 10, np.nan], "m") == "µm"
    assert suggest_length_unit_for_data("log", [1e-9, 10, np.inf], "m") == "m"
    assert suggest_length_unit_for_data("log", [1e-9, 10, np.inf, np.inf], "m") == "m"
    with pytest.raises(ValueError):
        suggest_length_unit_for_data("log", [np.nan, np.nan], "m")


def test_mangle_length_unit_to_ascii():
    assert mangle_length_unit_ascii("Å") == "A"


def test_mangle_length_unit_is_partially_idempotent():
    for unit in ["Å", "µm", "mm", "m"]:
        assert mangle_length_unit_utf8(mangle_length_unit_ascii(unit)) == unit


def test_ampere_is_not_angstrom():
    # By default, 'A' is the ASCII form of the ångström
    assert is_length_unit("A")
    assert mangle_length_unit_utf8("A") == "Å"
    # Formats that use 'A' for the ampere can switch this off
    assert not is_length_unit("A", ascii_angstrom=False)
    assert mangle_length_unit_utf8("A", ascii_angstrom=False) == "A"
    assert is_length_unit("Å", ascii_angstrom=False)
    assert mangle_length_unit_utf8("um", ascii_angstrom=False) == "µm"


length_exponents = {'Gm': 9, 'Mm': 6, 'km': 3, 'm': 0, 'mm': -3, 'µm': -6, 'um': -6, 'nm': -9, 'Å': -10, 'A': -10,
                    'pm': -12, 'fm': -15}


def test_conversion_factors_are_powers_of_ten():
    # Factors between length units are the floats closest to the exact
    # powers of ten, so that sizes survive a round trip through another unit
    for from_unit, from_exponent in length_exponents.items():
        for to_unit, to_exponent in length_exponents.items():
            assert get_unit_conversion_factor(from_unit, to_unit) == 10.0 ** (from_exponent - to_exponent)
    assert 500 * get_unit_conversion_factor('µm', 'mm') * get_unit_conversion_factor('mm', 'µm') == 500


def test_conversion_between_kinds():
    with pytest.raises(ValueError, match="'mV' is of kind voltage"):
        get_unit_conversion_factor('m', 'mV')
