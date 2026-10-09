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

import io
import os
import tempfile
from zipfile import ZipFile

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography import Topography, read_topography
from SurfaceTopography.IO import X3PReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_read(file_format_examples):
    surface = X3PReader(os.path.join(file_format_examples, 'x3p-1.x3p')).topography()
    nx, ny = surface.nb_grid_pts
    assert nx == 1035
    assert ny == 777
    sx, sy = surface.physical_sizes
    np.testing.assert_allclose(sx, 0.00068724, rtol=1e-5)
    np.testing.assert_allclose(sy, 0.00051593, rtol=1e-5)
    assert surface.unit == 'm'
    assert surface.is_uniform
    assert surface.has_undefined_data
    # Note: this file contains 1.7% undefined data points. The reference
    # value used to be 9.53e-05, which was an artifact of the mean height
    # being normalized by the total instead of the defined point count (the
    # heights sit at about -0.00572, so the resulting systematic offset
    # dwarfed the true roughness of this surface).
    np.testing.assert_allclose(surface.rms_height_from_area(), 2.3756839548083504e-07, rtol=1e-6)
    np.testing.assert_allclose(surface.interpolate_undefined_data().rms_gradient(), 0.15300264662900961, rtol=1e-6)
    assert surface.info['instrument']['name'] == 'Mountains Map Technology Software (DIGITAL SURF, version 6.2)'
    assert surface.info['instrument']['vendor'] == 'DIGITAL SURF'

    surface = X3PReader(os.path.join(file_format_examples, 'x3p-2.x3p')).topography()
    nx, ny = surface.nb_grid_pts
    assert nx == 650
    assert ny == 650
    sx, sy = surface.physical_sizes
    np.testing.assert_allclose(sx, 8.29767313942749e-05, rtol=1e-6)
    np.testing.assert_allclose(sy, 0.0002044783737930349, rtol=1e-6)
    assert surface.unit == 'm'
    assert surface.is_uniform
    assert not surface.has_undefined_data
    np.testing.assert_allclose(surface.rms_height_from_area(), 7.728033273597876e-08, rtol=1e-6)
    np.testing.assert_allclose(surface.rms_gradient(), 0.062070073998443276, rtol=1e-6)
    assert surface.info['instrument']['name'] == 'Mountains Map Technology Software (DIGITAL SURF, version 6.2)'
    assert surface.info['instrument']['vendor'] == 'DIGITAL SURF'

    surface = X3PReader(os.path.join(file_format_examples, 'x3p-3.x3p')).topography()
    nx, ny = surface.nb_grid_pts
    assert nx == 1199
    assert ny == 1199
    sx, sy = surface.physical_sizes
    np.testing.assert_allclose(sx, 0.0016148791409228245, rtol=1e-6)
    np.testing.assert_allclose(sy, 0.001612325270929275, rtol=1e-6)
    assert surface.unit == 'm'
    assert surface.is_uniform
    assert not surface.has_undefined_data
    np.testing.assert_allclose(surface.rms_height_from_area(), 3.6982281692457683e-06, rtol=1e-6)
    np.testing.assert_allclose(surface.rms_gradient(), 1.102796882522711, rtol=1e-6)
    assert surface.info['instrument']['name'] == 'NanoFocus AG'
    assert surface.info['instrument']['vendor'] == 'NanoFocus AG'

    surface = X3PReader(os.path.join(file_format_examples, 'x3p-4.x3p')).topography()
    nx, ny = surface.nb_grid_pts
    assert nx == 3427
    assert ny == 3463
    sx, sy = surface.physical_sizes
    np.testing.assert_allclose(sx, 0.004615672073346555, rtol=1e-6)
    np.testing.assert_allclose(sy, 0.004656782663242769, rtol=1e-6)
    assert surface.unit == 'm'
    assert surface.is_uniform
    assert surface.has_undefined_data
    np.testing.assert_allclose(surface.rms_height_from_area(), 3.6582125376441385e-06, rtol=1e-6)
    np.testing.assert_allclose(surface.interpolate_undefined_data().rms_gradient(), 1.124560711465191, rtol=1e-6)
    assert surface.info['instrument']['name'] == 'NanoFocus AG'
    assert surface.info['instrument']['vendor'] == 'NanoFocus AG'


def test_points_for_uniform_topography(file_format_examples):
    surface = X3PReader(os.path.join(file_format_examples, 'x3p-1.x3p')).topography()
    x, y, z = surface.positions_and_heights()
    np.testing.assert_allclose(np.mean(np.diff(x[:, 0])),
                               surface.physical_sizes[0] / surface.nb_grid_pts[0])
    np.testing.assert_allclose(np.mean(np.diff(y[0, :])),
                               surface.physical_sizes[1] / surface.nb_grid_pts[1])


# =============================================================================
# Writer tests
# =============================================================================

def test_write_x3p_roundtrip():
    """Test writing and reading back a topography preserves data."""
    np.random.seed(42)
    heights = np.random.randn(50, 60)
    t = Topography(heights, (1e-3, 1.2e-3), unit='m')

    with tempfile.NamedTemporaryFile(suffix='.x3p', delete=False) as f:
        fname = f.name

    try:
        t.to_x3p(fname)
        t2 = read_topography(fname)

        assert t2.nb_grid_pts == t.nb_grid_pts
        np.testing.assert_allclose(t2.physical_sizes, t.physical_sizes, rtol=1e-10)
        assert t2.unit == 'm'
        np.testing.assert_allclose(t2.heights(), t.heights(), rtol=1e-10)
    finally:
        os.unlink(fname)


def test_write_x3p_unit_conversion():
    """Test that unit conversion to meters works correctly."""
    np.random.seed(42)
    heights = np.random.randn(30, 40)
    t = Topography(heights, (100, 120), unit='um')

    with tempfile.NamedTemporaryFile(suffix='.x3p', delete=False) as f:
        fname = f.name

    try:
        t.to_x3p(fname)
        t2 = read_topography(fname)

        # X3P always stores in meters
        assert t2.unit == 'm'
        np.testing.assert_allclose(t2.physical_sizes, (100e-6, 120e-6), rtol=1e-10)

        # Heights should match when converted back
        np.testing.assert_allclose(t2.heights() * 1e6, t.heights(), rtol=1e-10)
    finally:
        os.unlink(fname)


@pytest.mark.parametrize('dtype', ['D', 'F'])
def test_write_x3p_dtypes(dtype):
    """Test different data types for height storage."""
    np.random.seed(42)
    heights = np.random.randn(20, 25)
    t = Topography(heights, (1e-3, 1.25e-3), unit='m')

    with tempfile.NamedTemporaryFile(suffix='.x3p', delete=False) as f:
        fname = f.name

    try:
        t.to_x3p(fname, dtype=dtype)
        t2 = read_topography(fname)

        assert t2.nb_grid_pts == t.nb_grid_pts

        # Float32 has lower precision
        if dtype == 'F':
            np.testing.assert_allclose(t2.heights(), t.heights(), rtol=1e-6)
        else:
            np.testing.assert_allclose(t2.heights(), t.heights(), rtol=1e-10)
    finally:
        os.unlink(fname)


def test_write_x3p_read_back_existing_file(file_format_examples):
    """Test that we can read, write, and read back an existing X3P file."""
    original = read_topography(os.path.join(file_format_examples, 'x3p-2.x3p'))

    with tempfile.NamedTemporaryFile(suffix='.x3p', delete=False) as f:
        fname = f.name

    try:
        original.to_x3p(fname)
        reread = read_topography(fname)

        assert reread.nb_grid_pts == original.nb_grid_pts
        np.testing.assert_allclose(reread.physical_sizes, original.physical_sizes, rtol=1e-10)
        np.testing.assert_allclose(reread.heights(), original.heights(), rtol=1e-10)
    finally:
        os.unlink(fname)


def test_write_x3p_1d_raises():
    """Test that writing 1D topography raises an error."""
    from SurfaceTopography import UniformLineScan
    t = UniformLineScan(np.random.randn(100), 1.0)

    with tempfile.NamedTemporaryFile(suffix='.x3p', delete=False) as f:
        fname = f.name

    try:
        with pytest.raises(ValueError, match="2D topographies"):
            t.to_x3p(fname)
    finally:
        if os.path.exists(fname):
            os.unlink(fname)


@pytest.mark.parametrize('dtype', ['I', 'L'])
def test_write_x3p_integer_dtypes(dtype):
    """Integer data is written as signed integers with increment and offset"""
    np.random.seed(42)
    heights = np.random.randn(20, 25)
    heights[3, 4] = np.nan
    t = Topography(np.ma.masked_invalid(heights), (1e-3, 1.25e-3), unit='m')

    buffer = io.BytesIO()
    t.to_x3p(buffer, dtype=dtype)
    buffer.seek(0)
    t2 = X3PReader(buffer).topography()
    assert t2.nb_grid_pts == t.nb_grid_pts
    np.testing.assert_array_equal(t2.heights().mask, np.isnan(heights))
    # Resolution of the integer representation
    tol = np.ptp(heights[~np.isnan(heights)]) / (2**16 if dtype == 'I' else 2**32)
    np.testing.assert_allclose(t2.heights()[~np.isnan(heights)], heights[~np.isnan(heights)], atol=tol)

    # Raw data uses the full signed range
    buffer.seek(0)
    with ZipFile(buffer) as z:
        raw = np.frombuffer(z.read('bindata/data.bin'), dtype='<i2' if dtype == 'I' else '<i4')
    assert raw.min() < 0


def _make_x3p(raw, dtype, increment=None, offset=None):
    """Minimal X3P with integer data; `raw` is in (ny, nx) order"""
    ny, nx = raw.shape
    z = f'<AxisType>A</AxisType><DataType>{dtype}</DataType>'
    if increment is not None:
        z += f'<Increment>{increment}</Increment>'
    if offset is not None:
        z += f'<Offset>{offset}</Offset>'
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<p:ISO5436_2 xmlns:p="http://www.opengps.eu/2008/ISO5436_2">
<Record1><Revision>ISO5436 - 2000</Revision><FeatureType>SUR</FeatureType><Axes>
<CX><AxisType>I</AxisType><DataType>D</DataType><Increment>1e-6</Increment><Offset>0</Offset></CX>
<CY><AxisType>I</AxisType><DataType>D</DataType><Increment>2e-6</Increment><Offset>0</Offset></CY>
<CZ>{z}</CZ></Axes></Record1>
<Record3><MatrixDimension><SizeX>{nx}</SizeX><SizeY>{ny}</SizeY><SizeZ>1</SizeZ></MatrixDimension>
<DataLink><PointDataLink>bindata/data.bin</PointDataLink></DataLink></Record3>
</p:ISO5436_2>"""
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w') as z:
        z.writestr('main.xml', xml)
        z.writestr('bindata/data.bin', raw.tobytes())
    buffer.seek(0)
    return buffer


@pytest.mark.parametrize('dtype, np_dtype', [('I', '<i2'), ('L', '<i4')])
def test_read_x3p_signed_integers(dtype, np_dtype):
    """ISO 5436-2 integer data types are signed"""
    raw = np.array([[-3, -1, 0], [1, 2, -32768]], dtype=np_dtype)
    t = X3PReader(_make_x3p(raw, dtype, increment=1e-9, offset=1e-6)).topography()
    assert t.nb_grid_pts == (3, 2)
    np.testing.assert_allclose(t.physical_sizes, (3e-6, 4e-6))
    np.testing.assert_allclose(t.heights(), raw.T * 1e-9 + 1e-6)
