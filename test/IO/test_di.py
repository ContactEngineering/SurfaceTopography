#
# Copyright 2020-2021, 2023 Lars Pastewka
#           2019-2020 Antoine Sanner
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

import datetime
import os
import pytest

import numpy as np

from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.IO import DIReader
from SurfaceTopography.Exceptions import CorruptFile

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_di_date(file_format_examples):
    t = read_topography(os.path.join(file_format_examples, 'di-1.di'))
    assert t.info['acquisition_time'] == str(datetime.datetime(2016, 1, 12, 9, 57, 48))
    assert t.info['instrument']['name'] == 'Dimension V'


def test_4byte_data(file_format_examples):
    r = DIReader(os.path.join(file_format_examples, 'di-5.di'))
    t = r.topography()
    np.testing.assert_allclose(t.rms_height_from_area(), 5.831926)
    assert t.info['instrument']['name'] == 'Dimension Icon'


def test_corrupted_file(file_format_examples):
    # Corruption should be detected when opening file; subsequent calls to `topography` must succeed
    with pytest.raises(CorruptFile):
        DIReader(os.path.join(file_format_examples, 'di_corrupted.di'))


def test_di7(file_format_examples):
    r = DIReader(os.path.join(file_format_examples, 'di-7.di'))
    r.topography()
