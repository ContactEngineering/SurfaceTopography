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

import h5py
import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography.IO import DATXReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")


def test_datx1_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'datx-1.datx')

    r = DATXReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 1000
    assert ny == 1000

    assert t.unit == 'nm'
    assert t.info['instrument']['vendor'] == 'Zygo'
    assert t.info['instrument']['serial'] == '87266'

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 6306280.26666992, rtol=1e-6)
    np.testing.assert_allclose(sy, 6306280.26666992, rtol=1e-6)

    np.testing.assert_allclose(t.max(), 13808.435547, rtol=1e-6)
    np.testing.assert_allclose(t.min(), -23393.847656, rtol=1e-6)

    np.testing.assert_allclose(t.rms_height_from_area(), 6304.986277, rtol=1e-6)


def test_datx2_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'datx-2.datx')

    r = DATXReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 1000
    assert ny == 1000

    assert t.unit == 'nm'
    assert t.info['instrument']['vendor'] == 'Zygo'
    assert t.info['instrument']['serial'] == '78137'

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 173108.716695, rtol=1e-6)
    np.testing.assert_allclose(sy, 173108.716695, rtol=1e-6)

    np.testing.assert_allclose(t.max(), 11.037771, rtol=1e-6)
    np.testing.assert_allclose(t.min(), -116.920823, rtol=1e-6)

    np.testing.assert_allclose(t.rms_height_from_area(), 1.9089580434424194, rtol=1e-6)


def _write_datx(file_path, data, dx, dy, no_data):
    """Write a minimal DATX file with a single surface dataset"""
    str_dtype = h5py.string_dtype()
    surface_path = '/Data/Surface/{00000000-0000-0000-0000-000000000001}'
    with h5py.File(file_path, 'w') as h5:
        metadata = np.array(
            [('Root', 'Measurement', '{M}'),
             ('{M}', 'Surface', '{S}'),
             ('{S}', 'Path', surface_path)],
            dtype=[('Source', str_dtype), ('Link', str_dtype), ('Destination', str_dtype)])
        h5.create_dataset('MetaData', data=metadata)
        dataset = h5.create_dataset(surface_path, data=data)
        converter_dtype = np.dtype([('Category', str_dtype), ('BaseUnit', str_dtype),
                                    ('Parameters', h5py.vlen_dtype(np.float64))])

        def converter(category, unit, parameters):
            c = np.empty(1, dtype=converter_dtype)
            c[0] = (category, unit, np.array(parameters, dtype=np.float64))
            return c

        dataset.attrs['X Converter'] = converter('LateralCat', 'Pixels', [0, dx, 0, 0])
        dataset.attrs['Y Converter'] = converter('LateralCat', 'Pixels', [0, dy, 0, 0])
        dataset.attrs['Z Converter'] = converter('HeightCat', 'NanoMeters', [0, 6e-7, 0.5, 1])
        dataset.attrs['Unit'] = np.array(['NanoMeters'], dtype=str_dtype)
        dataset.attrs['No Data'] = np.array([no_data])


def test_datx_orientation(tmp_path):
    """
    The surface dataset is stored row-major with rows running along y; the
    `X Converter` applies to the columns (as in Gwyddion's datxfile.c).
    """
    no_data = np.finfo(np.float64).max
    nb_rows, nb_columns = 3, 4
    data = np.arange(nb_rows * nb_columns, dtype=float).reshape(nb_rows, nb_columns)
    data[1, 2] = no_data
    data[2, 0] = np.nan
    file_path = tmp_path / 'synthetic.datx'
    _write_datx(file_path, data, dx=1e-6, dy=2e-6, no_data=no_data)

    t = DATXReader(file_path).topography()
    assert t.unit == 'nm'
    assert t.nb_grid_pts == (nb_columns, nb_rows)
    np.testing.assert_allclose(t.physical_sizes, (nb_columns * 1000, nb_rows * 2000))
    heights = t.heights()
    assert heights.shape == (nb_columns, nb_rows)
    # heights[x, y] == data[y, x]
    assert heights[3, 0] == data[0, 3]
    assert heights[1, 2] == data[2, 1]
    assert t.has_undefined_data
    mask = np.ma.getmaskarray(heights)
    assert mask[2, 1] and mask[0, 2]
    assert mask.sum() == 2
