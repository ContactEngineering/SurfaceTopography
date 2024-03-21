#
# Copyright 2020-2023 Lars Pastewka
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

import os

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography.IO import XYZReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


@pytest.mark.parametrize('mode,encoding', [('rb', None), ('r', 'utf-8'), ('r', 'latin-1')])
def test_dektak_csv(file_format_examples, mode, encoding):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, 'dektak-1.csv')

    r = XYZReader(open(file_path, mode=mode, encoding=encoding))

    t = r.topography()

    assert not t.is_uniform
    np.testing.assert_allclose(t.rms_height_from_profile(), 4.763596)
    np.testing.assert_allclose(t.rms_slope_from_profile(), 0.015448, rtol=1e-5)

    assert t.info['instrument'] == {
        'parameters': {
            'tip_radius': {
                'value': 2.5,
                'unit': 'µm',
            }
        }
    }

    np.testing.assert_allclose(t.short_reliability_cutoff(), 1.57, rtol=0.01)


@pytest.mark.parametrize('fn', ['dektak-1.csv', 'csv-1.csv', 'csv-2.csv', 'csv-3.csv'])
def test_generic_csv(file_format_examples, fn):
    file_path = os.path.join(file_format_examples, fn)

    XYZReader(file_path)
