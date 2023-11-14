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

from NuMPI import MPI

from SurfaceTopography.Support.UnitConversion import suggest_length_unit, suggest_length_unit_for_data

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_suggest_length_unit():
    assert suggest_length_unit('log', 1e-3, 1e-2) == 'mm'
    assert suggest_length_unit('log', 1e-6, 1e-5) == 'µm'

    assert suggest_length_unit('linear', 0, 1e-2) == 'mm'

    assert suggest_length_unit('log', 1e-9, 1) == 'µm'
    assert suggest_length_unit('log', 1e-9, 10) == 'µm'
    assert suggest_length_unit('log', 1e-9, 100) == 'µm'
    assert suggest_length_unit('log', 1e-9, 100) == 'µm'
    assert suggest_length_unit('log', 1e-9, 10000) == 'mm'
    assert suggest_length_unit('log', 1e-9, 100000) == 'mm'


def test_unit_outside_range():
    assert suggest_length_unit('log', 1e-21, 1e-18) == 'fm'
    assert suggest_length_unit('log', 1e12, 1e15) == 'Gm'


def test_nan_and_inf():
    assert suggest_length_unit_for_data('log', [1e-9, 10], 'm') == 'µm'
    assert suggest_length_unit_for_data('log', [1e-9, 10, np.NaN], 'm') == 'µm'
    assert suggest_length_unit_for_data('log', [1e-9, 10, np.Inf], 'm') == 'm'
    assert suggest_length_unit_for_data('log', [1e-9, 10, np.Inf, np.Inf], 'm') == 'm'
    with pytest.raises(ValueError):
        suggest_length_unit_for_data('log', [np.NaN, np.NaN], 'm')
