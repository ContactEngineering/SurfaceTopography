#
# Copyright 2026 Lars Pastewka
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

"""
Tests for scan line alignment functionality.
"""

import numpy as np
import pytest

from SurfaceTopography import Topography


def test_scan_line_align_removes_offsets():
    """Test that scan line alignment removes inter-line offsets."""
    # Create topography with artificial scan line offsets
    nx, ny = 64, 64
    h = np.zeros((nx, ny))
    # Add random offset to each line
    np.random.seed(42)
    offsets = np.random.randn(nx) * 10
    for i in range(nx):
        h[i, :] += offsets[i]

    topo = Topography(h, (1, 1), unit='um')
    aligned = topo.scan_line_align()

    # Check that aligned heights have reduced line-to-line variation
    aligned_heights = aligned.heights()
    line_means = [aligned_heights[i, :].mean() for i in range(nx)]
    # Standard deviation of line means should be very small after alignment
    assert np.std(line_means) < 0.01


def test_scan_line_align_removes_tilt():
    """Test that scan line alignment removes per-line tilt."""
    nx, ny = 32, 32
    y = np.arange(ny) / ny
    h = np.zeros((nx, ny))
    # Add different tilt to each line
    np.random.seed(42)
    for i in range(nx):
        slope = np.random.randn() * 5
        h[i, :] = slope * y

    topo = Topography(h, (1, 1), unit='um')
    aligned = topo.scan_line_align()

    # Check that slopes within lines are reduced
    aligned_heights = aligned.heights()
    t = np.arange(ny)
    for i in range(nx):
        slope = np.polyfit(t, aligned_heights[i, :], 1)[0]
        assert abs(slope) < 0.001


def test_scan_line_align_y_direction():
    """Test scan line alignment in y-direction."""
    nx, ny = 32, 32
    h = np.zeros((nx, ny))
    # Add offsets to columns instead of rows
    np.random.seed(42)
    for j in range(ny):
        h[:, j] += np.random.randn() * 5

    topo = Topography(h, (1, 1), unit='um')
    aligned = topo.scan_line_align(direction='y')

    aligned_heights = aligned.heights()
    col_means = [aligned_heights[:, j].mean() for j in range(ny)]
    assert np.std(col_means) < 0.01


def test_scan_line_align_preserves_features():
    """Test that alignment preserves actual surface features."""
    nx, ny = 64, 64
    x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny),
                       indexing='ij')
    # Create surface with a Gaussian bump
    h = 10 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.05)
    # Add scan line artifacts
    np.random.seed(42)
    for i in range(nx):
        h[i, :] += np.random.randn() * 0.5  # Offset
        h[i, :] += np.random.randn() * 0.1 * np.arange(ny) / ny  # Tilt

    topo = Topography(h, (1, 1), unit='um')
    aligned = topo.scan_line_align()

    # The bump should still be visible after alignment
    # (center should be higher than corners)
    aligned_heights = aligned.heights()
    center_height = aligned_heights[nx // 2, ny // 2]
    corner_heights = [aligned_heights[0, 0], aligned_heights[0, -1],
                      aligned_heights[-1, 0], aligned_heights[-1, -1]]
    assert center_height > np.mean(corner_heights) + 5


def test_scan_line_align_masked_data():
    """Test scan line alignment with masked data."""
    nx, ny = 32, 32
    np.random.seed(42)
    h = np.random.randn(nx, ny)
    # Add scan line offsets
    for i in range(nx):
        h[i, :] += i * 0.5

    # Mask some data
    mask = np.zeros((nx, ny), dtype=bool)
    mask[5:10, 10:20] = True
    h = np.ma.array(h, mask=mask)

    topo = Topography(h, (1, 1), unit='um')
    aligned = topo.scan_line_align()

    # Should complete without error
    aligned_heights = aligned.heights()
    assert aligned_heights.shape == (nx, ny)


def test_scan_line_align_pipeline():
    """Test scan line alignment in pipeline with other operations."""
    np.random.seed(42)
    h = np.random.randn(32, 32)
    # Add a global tilt
    x, y = np.meshgrid(np.arange(32) / 32, np.arange(32) / 32, indexing='ij')
    h += 5 * x + 3 * y
    # Add scan line offsets
    for i in range(32):
        h[i, :] += np.random.randn() * 2

    topo = Topography(h, (1, 1), unit='um')

    # Should work in pipeline: first align scan lines, then global detrend
    result = topo.scan_line_align().detrend()
    assert result.heights().shape == (32, 32)

    # The pipeline should be preserved
    assert len(result.pipeline()) >= 2


def test_scan_line_align_mode_mean():
    """Test scan line alignment with mean mode."""
    nx, ny = 32, 32
    np.random.seed(42)
    h = np.zeros((nx, ny))
    for i in range(nx):
        h[i, :] += np.random.randn() * 5

    topo = Topography(h, (1, 1), unit='um')
    aligned = topo.scan_line_align(mode='mean')

    aligned_heights = aligned.heights()
    line_means = [aligned_heights[i, :].mean() for i in range(nx)]
    assert np.std(line_means) < 0.01


def test_scan_line_align_periodicity():
    """Test that scan line alignment sets is_periodic to False."""
    h = np.random.randn(16, 16)
    topo = Topography(h, (1, 1), unit='um', periodic=True)
    assert topo.is_periodic

    aligned = topo.scan_line_align()
    assert not aligned.is_periodic


def test_scan_line_align_coefficients():
    """Test that line coefficients and offsets are accessible."""
    nx, ny = 16, 16
    np.random.seed(42)
    h = np.zeros((nx, ny))
    for i in range(nx):
        h[i, :] = 2 + 3 * np.arange(ny) / ny  # Same tilt for all lines

    topo = Topography(h, (1, 1), unit='um')
    aligned = topo.scan_line_align()

    # Access coefficients
    coeffs = aligned.line_coeffs
    assert coeffs.shape == (nx, 2)
    # All lines should have similar tilt coefficient
    assert np.allclose(coeffs[:, 1], coeffs[0, 1], rtol=0.01)

    # Access offsets
    offsets = aligned.offsets
    assert offsets.shape == (nx,)


def test_scan_line_align_1d_raises():
    """Test that scan line alignment raises for 1D line scans."""
    from SurfaceTopography import UniformLineScan

    h = np.random.randn(64)
    line_scan = UniformLineScan(h, (1,), unit='um')

    with pytest.raises(ValueError, match="only supported for 2D"):
        line_scan.scan_line_align()


def test_scan_line_align_pickling():
    """Test that aligned topography can be pickled and unpickled."""
    import pickle

    np.random.seed(42)
    h = np.random.randn(16, 16)
    for i in range(16):
        h[i, :] += i * 0.5

    topo = Topography(h, (1, 1), unit='um')
    aligned = topo.scan_line_align()

    # Trigger computation
    original_heights = aligned.heights()

    # Pickle and unpickle
    pickled = pickle.dumps(aligned)
    unpickled = pickle.loads(pickled)

    # Should produce same heights
    np.testing.assert_array_almost_equal(original_heights, unpickled.heights())
    assert unpickled.direction == aligned.direction
    assert unpickled.mode == aligned.mode


def test_scan_line_align_degree_0():
    """Test scan line alignment with degree 0 (offset only)."""
    nx, ny = 32, 32
    np.random.seed(42)
    h = np.zeros((nx, ny))
    # Add random offset to each line (no tilt)
    offsets = np.random.randn(nx) * 10
    for i in range(nx):
        h[i, :] += offsets[i]
    # Add a linear tilt that should NOT be removed with degree=0
    t = np.arange(ny) / ny
    for i in range(nx):
        h[i, :] += 5 * t

    topo = Topography(h, (1, 1), unit='um')
    aligned = topo.scan_line_align(degree=0)

    # Offsets should be removed
    aligned_heights = aligned.heights()
    line_means = [aligned_heights[i, :].mean() for i in range(nx)]
    assert np.std(line_means) < 0.01

    # But the tilt should still be present (same for all lines)
    slopes = [np.polyfit(np.arange(ny), aligned_heights[i, :], 1)[0]
              for i in range(nx)]
    assert np.allclose(slopes, slopes[0], rtol=0.01)
    assert abs(slopes[0]) > 0.1  # Tilt should still be there

    # Check coefficients shape
    assert aligned.line_coeffs.shape == (nx, 1)
    assert aligned.degree == 0


def test_scan_line_align_degree_2():
    """Test scan line alignment with degree 2 (quadratic/curvature)."""
    nx, ny = 32, 64
    t = np.arange(ny) / ny
    h = np.zeros((nx, ny))
    # Add quadratic (parabolic) curvature to each line - scanner bow artifact
    np.random.seed(42)
    for i in range(nx):
        a = np.random.randn() * 2  # Curvature coefficient
        b = np.random.randn() * 3  # Tilt coefficient
        c = np.random.randn() * 5  # Offset
        h[i, :] = a * t**2 + b * t + c

    topo = Topography(h, (1, 1), unit='um')

    # With degree=2, quadratic term should be removed
    aligned_deg2 = topo.scan_line_align(degree=2)
    aligned_heights_2 = aligned_deg2.heights()
    # Check residual curvature is small
    curvatures_2 = [np.polyfit(t, aligned_heights_2[i, :], 2)[0]
                    for i in range(nx)]
    # All curvatures should be near zero after degree-2 correction
    assert all(abs(c) < 0.01 for c in curvatures_2)

    # Check coefficients shape
    assert aligned_deg2.line_coeffs.shape == (nx, 3)
    assert aligned_deg2.degree == 2

    # The original coefficients should be recovered (approximately)
    # Note: numpy polyfit returns highest power first
    original_curvatures = [np.polyfit(t, h[i, :], 2)[0] for i in range(nx)]
    fitted_curvatures = aligned_deg2.line_coeffs[:, 0]  # Highest power first
    np.testing.assert_allclose(original_curvatures, fitted_curvatures, rtol=0.01)


def test_scan_line_align_degree_3():
    """Test scan line alignment with degree 3 (cubic)."""
    nx, ny = 16, 64
    t = np.arange(ny) / ny
    h = np.zeros((nx, ny))
    # Add cubic polynomial to each line
    np.random.seed(42)
    for i in range(nx):
        a = np.random.randn() * 1
        b = np.random.randn() * 2
        c = np.random.randn() * 3
        d = np.random.randn() * 5
        h[i, :] = a * t**3 + b * t**2 + c * t + d

    topo = Topography(h, (1, 1), unit='um')
    aligned = topo.scan_line_align(degree=3)

    # Check coefficients shape
    assert aligned.line_coeffs.shape == (nx, 4)
    assert aligned.degree == 3

    # Residuals should be small
    aligned_heights = aligned.heights()
    for i in range(nx):
        residual = np.polyfit(np.arange(ny), aligned_heights[i, :], 3)
        # All polynomial coefficients should be near zero
        assert np.allclose(residual[:-1], 0, atol=0.01)


def test_scan_line_align_negative_degree_raises():
    """Test that negative degree raises ValueError."""
    h = np.random.randn(16, 16)
    topo = Topography(h, (1, 1), unit='um')

    with pytest.raises(ValueError, match="non-negative"):
        topo.scan_line_align(degree=-1)


def test_scan_line_align_pickling_with_degree():
    """Test that aligned topography with custom degree can be pickled."""
    import pickle

    np.random.seed(42)
    nx, ny = 16, 32
    t = np.arange(ny) / ny
    h = np.zeros((nx, ny))
    for i in range(nx):
        h[i, :] = np.random.randn() * t**2 + np.random.randn() * t + i * 0.5

    topo = Topography(h, (1, 1), unit='um')
    aligned = topo.scan_line_align(degree=2)

    # Trigger computation
    original_heights = aligned.heights()

    # Pickle and unpickle
    pickled = pickle.dumps(aligned)
    unpickled = pickle.loads(pickled)

    # Should produce same heights
    np.testing.assert_array_almost_equal(original_heights, unpickled.heights())
    assert unpickled.degree == 2
    assert unpickled.line_coeffs.shape == (nx, 3)


def test_scan_line_align_scanner_bow_correction():
    """Test realistic scanner bow correction with degree=2."""
    nx, ny = 64, 128
    t = np.linspace(0, 1, ny)
    h = np.zeros((nx, ny))

    # Simulate scanner bow: parabolic distortion that varies slightly per line
    np.random.seed(42)
    base_bow = 2.0  # Base bow amplitude
    for i in range(nx):
        bow = base_bow + np.random.randn() * 0.2  # Slight variation
        # Parabolic bow: maximum deviation at center
        h[i, :] = bow * 4 * (t - 0.5)**2

    # Add some actual surface features
    x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing='ij')
    features = 0.5 * np.sin(2 * np.pi * x * 3) * np.sin(2 * np.pi * y * 3)
    h += features

    topo = Topography(h, (10, 20), unit='um')

    # Correct with degree=2
    aligned = topo.scan_line_align(degree=2)
    aligned_heights = aligned.heights()

    # The parabolic bow should be removed
    for i in range(nx):
        coeffs = np.polyfit(t, aligned_heights[i, :], 2)
        # Quadratic coefficient should be near zero
        assert abs(coeffs[0]) < 0.1

    # The sinusoidal features should still be present (approximately)
    # Check that there's still some variation consistent with the features
    assert aligned_heights.std() > 0.1
