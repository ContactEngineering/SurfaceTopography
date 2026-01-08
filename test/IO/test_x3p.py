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
import tempfile

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
    np.testing.assert_allclose(surface.rms_height_from_area(), 9.528212249587946e-05, rtol=1e-6)
    np.testing.assert_allclose(surface.interpolate_undefined_data().rms_gradient(), 0.15300265543799388, rtol=1e-6)
    assert surface.info['instrument']['name'] == 'Mountains Map Technology Software (DIGITAL SURF, version 6.2)'

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
