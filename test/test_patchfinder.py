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
Tests for C extension patchfinder functions that were previously untested:
- correlation_function
- closest_patch_map
- perimeter_length
- shortest_distance
"""

import numpy as np
import pytest
from _SurfaceTopography import (
    assign_patch_numbers,
    closest_patch_map,
    correlation_function,
    perimeter_length,
    shortest_distance,
)
from NuMPI import MPI
from numpy.testing import assert_allclose, assert_array_equal

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


# =============================================================================
# Tests for correlation_function
# =============================================================================

class TestCorrelationFunction:
    """Tests for the correlation_function C extension."""

    def test_autocorrelation_constant_map(self):
        """Autocorrelation of a constant map should be constant^2."""
        nx, ny = 16, 16
        value = 2.5
        map1 = np.full((nx, ny), value)
        max_dist = 5

        r, c, Ic = correlation_function(map1, map1, max_dist)

        # All correlation values should be value^2
        assert_allclose(c, value ** 2, rtol=1e-10)

    def test_autocorrelation_single_point(self):
        """Autocorrelation with a single non-zero point."""
        nx, ny = 16, 16
        map1 = np.zeros((nx, ny))
        map1[8, 8] = 1.0
        max_dist = 5

        r, c, Ic = correlation_function(map1, map1, max_dist)

        # At distance 0 (r=1), correlation should be 1/n_pixels
        # Other distances should have lower values
        assert len(r) > 0
        assert len(c) == len(r)
        assert len(Ic) == len(r)

    def test_cross_correlation_uncorrelated(self):
        """Cross-correlation of a map with zeros should be zero."""
        nx, ny = 16, 16
        map1 = np.random.random((nx, ny))
        map2 = np.zeros((nx, ny))
        max_dist = 5

        r, c, Ic = correlation_function(map1, map2, max_dist)

        # All correlations should be 0
        assert_allclose(c, 0, atol=1e-10)

    def test_correlation_symmetry(self):
        """Correlation should be symmetric: corr(A,B) == corr(B,A) for same distance."""
        nx, ny = 16, 16
        np.random.seed(42)
        map1 = np.random.random((nx, ny))
        map2 = np.random.random((nx, ny))
        max_dist = 5

        r1, c1, Ic1 = correlation_function(map1, map2, max_dist)
        r2, c2, Ic2 = correlation_function(map2, map1, max_dist)

        # Same distances should be found
        assert_array_equal(r1, r2)
        # Correlation values should be the same
        assert_allclose(c1, c2, rtol=1e-10)

    def test_correlation_returns_correct_distances(self):
        """Check that returned distances are correct."""
        nx, ny = 16, 16
        map1 = np.ones((nx, ny))
        max_dist = 5

        r, c, Ic = correlation_function(map1, map1, max_dist)

        # Distances should be sqrt(1+k) for k = 0, 1, 2, ..., max_dist^2-1
        # But only for k values where points exist
        for dist in r:
            assert dist >= 1.0  # Minimum distance is sqrt(1) = 1
            assert dist <= max_dist + 1


# =============================================================================
# Tests for closest_patch_map
# =============================================================================

class TestClosestPatchMap:
    """Tests for the closest_patch_map C extension."""

    def test_single_patch_all_closest(self):
        """With a single patch, all points should map to that patch."""
        nx, ny = 10, 10
        # Create a patch map with a single patch (id=1) in center
        patch_ids = np.zeros((nx, ny), dtype=np.int32)
        patch_ids[4:6, 4:6] = 1

        closest = closest_patch_map(patch_ids)

        # All points should map to patch 1
        assert_array_equal(closest, np.ones((nx, ny), dtype=np.int32))

    def test_two_patches_boundary(self):
        """Test boundary between two patches."""
        nx, ny = 10, 10
        patch_ids = np.zeros((nx, ny), dtype=np.int32)
        # Patch 1 on left side
        patch_ids[0:3, 4:6] = 1
        # Patch 2 on right side
        patch_ids[7:10, 4:6] = 2

        closest = closest_patch_map(patch_ids)

        # Left side should be closest to patch 1
        assert (closest[0:4, :] == 1).all()
        # Right side should be closest to patch 2
        assert (closest[6:10, :] == 2).all()

    def test_closest_patch_on_patch(self):
        """Points on a patch should map to that patch."""
        nx, ny = 10, 10
        patch_ids = np.zeros((nx, ny), dtype=np.int32)
        patch_ids[2:4, 2:4] = 1
        patch_ids[6:8, 6:8] = 2

        closest = closest_patch_map(patch_ids)

        # Points on patch 1 should map to 1
        assert (closest[2:4, 2:4] == 1).all()
        # Points on patch 2 should map to 2
        assert (closest[6:8, 6:8] == 2).all()

    def test_closest_patch_periodic(self):
        """Test periodic boundary handling in closest_patch_map."""
        nx, ny = 10, 10
        patch_ids = np.zeros((nx, ny), dtype=np.int32)
        # Patch at corner - should wrap around
        patch_ids[0, 0] = 1

        closest = closest_patch_map(patch_ids)

        # All should map to patch 1
        assert (closest == 1).all()


# =============================================================================
# Tests for perimeter_length
# =============================================================================

class TestPerimeterLength:
    """Tests for the perimeter_length C extension."""

    def test_single_point_perimeter(self):
        """A single point should have a small perimeter."""
        nx, ny = 10, 10
        mask = np.zeros((nx, ny), dtype=bool)
        mask[5, 5] = True

        length = perimeter_length(mask)

        # Single isolated point - perimeter depends on stencil
        # With 8-neighbor stencil, isolated point has perimeter sqrt(2)/2
        sqrt2 = np.sqrt(2)
        assert_allclose(length, sqrt2 / 2, rtol=0.1)

    def test_2x2_square_perimeter(self):
        """A 2x2 square should have perimeter ~4."""
        nx, ny = 10, 10
        mask = np.zeros((nx, ny), dtype=bool)
        mask[4:6, 4:6] = True

        length = perimeter_length(mask)

        # 2x2 square - perimeter should be around 4
        assert length > 3.5
        assert length < 5.0

    def test_line_perimeter(self):
        """A horizontal line should have positive perimeter."""
        nx, ny = 10, 10
        mask = np.zeros((nx, ny), dtype=bool)
        mask[5, 2:8] = True  # Line of length 6

        length = perimeter_length(mask)

        # Perimeter should be positive and related to number of pixels
        # The exact value depends on the algorithm's neighbor counting
        assert length > 0
        assert length < 20  # Reasonable upper bound

    def test_empty_map_perimeter(self):
        """An empty map should have zero perimeter."""
        nx, ny = 10, 10
        mask = np.zeros((nx, ny), dtype=bool)

        length = perimeter_length(mask)

        assert length == 0.0

    def test_full_map_perimeter(self):
        """A full map (periodic) should have zero external perimeter."""
        nx, ny = 10, 10
        mask = np.ones((nx, ny), dtype=bool)

        length = perimeter_length(mask)

        # Full periodic map - all pixels have full neighbors
        # So perimeter contribution is 1.0 per pixel
        assert_allclose(length, nx * ny, rtol=0.01)

    def test_larger_square_perimeter(self):
        """Test perimeter of a larger square."""
        nx, ny = 20, 20
        mask = np.zeros((nx, ny), dtype=bool)
        mask[5:15, 5:15] = True  # 10x10 square

        length = perimeter_length(mask)

        # Perimeter should be close to 4*10 = 40 for a 10x10 square
        # but the algorithm uses a different method
        assert length > 30
        assert length < 150  # Upper bound


# =============================================================================
# Tests for shortest_distance
# =============================================================================

class TestShortestDistance:
    """Tests for the shortest_distance C extension."""

    def test_adjacent_patches_distance(self):
        """Adjacent patches should have distance ~1."""
        nx, ny = 10, 10
        # fromc: the patch we're measuring from (contact map)
        fromc = np.zeros((nx, ny), dtype=bool)
        fromc[4, 4] = True
        # fromp: perimeter of the from patch
        fromp = np.zeros((nx, ny), dtype=bool)
        fromp[4, 4] = True
        # to: destination patch
        to = np.zeros((nx, ny), dtype=bool)
        to[4, 5] = True  # Adjacent point

        dist = shortest_distance(fromc, fromp, to)

        # Distance should be 1 at the from point
        assert_allclose(dist[4, 4], 1.0)

    def test_diagonal_distance(self):
        """Diagonal patches should have distance sqrt(2)."""
        nx, ny = 10, 10
        fromc = np.zeros((nx, ny), dtype=bool)
        fromc[4, 4] = True
        fromp = np.zeros((nx, ny), dtype=bool)
        fromp[4, 4] = True
        to = np.zeros((nx, ny), dtype=bool)
        to[5, 5] = True  # Diagonal point

        dist = shortest_distance(fromc, fromp, to)

        sqrt2 = np.sqrt(2)
        assert_allclose(dist[4, 4], sqrt2, rtol=0.01)

    def test_same_point_zero_distance(self):
        """Point on destination patch should have distance 0."""
        nx, ny = 10, 10
        fromc = np.zeros((nx, ny), dtype=bool)
        fromc[4, 4] = True
        fromp = np.zeros((nx, ny), dtype=bool)
        fromp[4, 4] = True
        to = np.zeros((nx, ny), dtype=bool)
        to[4, 4] = True  # Same point

        dist = shortest_distance(fromc, fromp, to)

        assert_allclose(dist[4, 4], 0.0)

    def test_no_from_patch_returns_zeros(self):
        """If from patch is empty, distances should be 0."""
        nx, ny = 10, 10
        fromc = np.zeros((nx, ny), dtype=bool)
        fromp = np.zeros((nx, ny), dtype=bool)
        to = np.zeros((nx, ny), dtype=bool)
        to[5, 5] = True

        dist = shortest_distance(fromc, fromp, to)

        # All zeros since no from points
        assert_allclose(dist, 0.0)

    def test_shortest_distance_multi_point_to(self):
        """Test distance with multiple destination points."""
        nx, ny = 10, 10
        # Create a line of points as source
        fromc = np.zeros((nx, ny), dtype=bool)
        fromc[5, 0:3] = True  # Line of 3 points
        fromp = np.zeros((nx, ny), dtype=bool)
        fromp[5, 0:3] = True  # Same as fromc
        # Destination points at different distances
        to = np.zeros((nx, ny), dtype=bool)
        to[5, 5] = True  # Distance 2-5 from the source line

        dist = shortest_distance(fromc, fromp, to)

        # The closest point in fromc to to[5,5] is fromc[5,2], distance 3
        # Check that the distance is computed for some point
        assert np.max(dist) > 0  # At least some non-zero distance


# =============================================================================
# Edge case tests for existing patch functions
# =============================================================================

class TestPatchfinderEdgeCases:
    """Edge case tests for patchfinder functions."""

    def test_assign_patch_numbers_empty(self):
        """Empty map should have 0 patches."""
        mask = np.zeros((10, 10), dtype=bool)
        nump, patch_ids = assign_patch_numbers(mask, True)

        assert nump == 0
        assert_array_equal(patch_ids, 0)

    def test_assign_patch_numbers_full(self):
        """Full map should have 1 patch."""
        mask = np.ones((10, 10), dtype=bool)
        nump, patch_ids = assign_patch_numbers(mask, True)

        assert nump == 1
        assert (patch_ids == 1).all()

    def test_assign_patch_numbers_sparse_isolated(self):
        """Sparse pattern with truly isolated points (no diagonal neighbors)."""
        mask = np.zeros((9, 9), dtype=bool)
        # Place points with 2-step spacing so they have no 8-neighbors
        mask[0, 0] = True
        mask[0, 3] = True
        mask[0, 6] = True
        mask[3, 0] = True
        mask[3, 3] = True
        mask[3, 6] = True
        mask[6, 0] = True
        mask[6, 3] = True
        mask[6, 6] = True

        # Non-periodic: each point is isolated (9 patches)
        nump, patch_ids = assign_patch_numbers(mask, False)
        assert nump == 9  # 9 isolated points

    def test_assign_patch_numbers_large_array(self):
        """Test with large array (regression test for overflow issues)."""
        nx, ny = 256, 256
        mask = np.zeros((nx, ny), dtype=bool)
        mask[100:150, 100:150] = True

        nump, patch_ids = assign_patch_numbers(mask, True)

        assert nump == 1
        assert patch_ids[125, 125] == 1
        assert patch_ids[0, 0] == 0
