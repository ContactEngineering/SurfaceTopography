#
# Copyright 2020-2021, 2023 Lars Pastewka
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

import h5py
import numpy as np
import scipy.io
import pytest

from NuMPI import MPI

from SurfaceTopography.IO import MatReader, detect_format, open_topography

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_read(file_format_examples):
    reader = MatReader(os.path.join(file_format_examples, 'mat-1.mat'))
    nx, ny = reader.channels[0].nb_grid_pts
    assert nx == 2048
    assert ny == 2048

    topography = reader.topography(physical_sizes=[1., 1.])
    nx, ny = topography.nb_grid_pts
    assert nx == 2048
    assert ny == 2048
    np.testing.assert_allclose(topography.rms_height_from_area(), 1.234061e-07, rtol=1e-6)
    assert topography.is_uniform


def test_ignore_non_numeric_variables(tmp_path):
    # Only two-dimensional numerical matrices are topographies; structures,
    # strings, cell arrays, empty and three-dimensional arrays are ignored
    fn = str(tmp_path / 'v5.mat')
    heights = np.arange(12.).reshape(3, 4)
    scipy.io.savemat(fn, dict(heights=heights, s=dict(a=1), c='hello',
                              cell=np.array([[1, 'a']], dtype=object),
                              volume=np.ones((2, 3, 4)), empty=np.zeros((0, 0)),
                              mask=np.ones((3, 4), dtype=bool)))
    reader = MatReader(fn)
    assert sorted(c.name for c in reader.channels) == ['heights', 'mask']
    channel = [c for c in reader.channels if c.name == 'heights'][0]
    assert channel.nb_grid_pts == (3, 4)
    t = reader.topography(channel_index=channel.index, physical_sizes=(1., 1.))
    np.testing.assert_allclose(t.heights(), heights)


def _write_mat73(fn, heights):
    """
    Write a minimal version 7.3 MAT-file: a HDF5 file with a 512 byte
    MATLAB header (user block). MATLAB stores matrices in column-major order,
    i.e. the HDF5 dataset holds the transpose of the matrix.
    """
    with h5py.File(fn, 'w', userblock_size=512) as f:
        d = f.create_dataset('heights', data=heights.T)
        d.attrs['MATLAB_class'] = np.bytes_('double')
        d = f.create_dataset('label', data=np.array([[104], [105]], dtype=np.uint16))
        d.attrs['MATLAB_class'] = np.bytes_('char')
        g = f.create_group('s')
        g.attrs['MATLAB_class'] = np.bytes_('struct')
        g.create_dataset('a', data=np.ones((2, 2)))
        d = f.create_dataset('empty', data=np.array([0, 0], dtype=np.uint64))
        d.attrs['MATLAB_class'] = np.bytes_('double')
        d.attrs['MATLAB_empty'] = np.uint8(1)
        d = f.create_dataset('z', data=np.zeros((4, 3), dtype=[('real', '<f8'), ('imag', '<f8')]))
        d.attrs['MATLAB_class'] = np.bytes_('double')
    text = b'MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: Thu Oct  8 12:00:00 2026 HDF5 schema 1.00 .'
    header = text.ljust(116, b' ') + b'\x00' * 8 + b'\x00\x02' + b'IM'
    with open(fn, 'r+b') as f:
        f.write(header)


def test_read_mat73(tmp_path):
    fn = str(tmp_path / 'v73.mat')
    heights = np.arange(12.).reshape(3, 4)
    _write_mat73(fn, heights)

    reader = MatReader(fn)
    assert [c.name for c in reader.channels] == ['heights']
    assert reader.channels[0].nb_grid_pts == (3, 4)
    t = reader.topography(physical_sizes=(1., 1.))
    # Same orientation as for version 5 files
    np.testing.assert_allclose(t.heights(), heights)

    assert detect_format(fn) == 'mat'
    with open(fn, 'rb') as f:
        reader = open_topography(f)
        assert reader.format() == 'mat'
        np.testing.assert_allclose(reader.topography(physical_sizes=(1., 1.)).heights(), heights)
