#
# Copyright 2021, 2023 Lars Pastewka
#           2021 Michael Röttger
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

from SurfaceTopography.Exceptions import UndefinedDataError
from SurfaceTopography.IO import open_topography, read_topography

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


@pytest.fixture
def make_topography_with_missing_data(file_format_examples):
    """Returns a topography with missing data, optionally scaled.

    Returns
    -------
    Factory function for a topography with missing data.
    """

    def _make_topo(dim, is_nonuniform, height_scale_factor=None):
        """
        Parameters
        ----------

        dim: int
            Give 1 or 2 for 1D or 2D
        is_nonuniform: bool
        height_scale_factor: float or None
            If given, the topography will be scaled by this factor

        Returns
        -------
        Topography with missing data.
        """
        if dim == 2:
            if is_nonuniform:
                raise NotImplementedError("Test not implemented for 2D nonuniform topographies.")
            fn = 'xyz-2.txt'
        elif dim == 1:
            if is_nonuniform:
                fn = 'xy-6.txt'
            else:
                fn = 'xy-5.txt'

        r = open_topography(os.path.join(file_format_examples, fn))
        topo_kwargs = {}
        if height_scale_factor is not None:
            topo_kwargs['height_scale_factor'] = height_scale_factor
        return r.topography(**topo_kwargs)

    return _make_topo


@pytest.mark.parametrize("dim, is_nonuniform, height_scale_factor, detrend_mode", [
    (1, False, None, None),  # None means: do not apply
    (1, False, 2, None),
    (1, False, None, 'center'),
    (1, False, 2, 'center'),
    # 1D nonuniform data with missing points is not supported at the moment
    # (1, True, None, None),  # None means: do not apply
    # (1, True, 2, None),
    # (1, True, None, 'center'),
    # (1, True, 2, 'center'),
    (2, False, None, None),  # None means: do not apply
    (2, False, 2, None),
    (2, False, None, 'center'),
    (2, False, 2, 'center'),
])
def test_masked_array_kept_by_netcdf(make_topography_with_missing_data, dim, is_nonuniform,
                                     height_scale_factor, detrend_mode):
    """For different ways in the pipeline .to_netcdf() should keep masked data."""
    topo = make_topography_with_missing_data(dim=dim, is_nonuniform=is_nonuniform,
                                             height_scale_factor=height_scale_factor)
    assert np.ma.is_masked(topo.heights()), "Masked array already lost before detrending"

    if detrend_mode is not None:
        topo = topo.detrend(detrend_mode)

    assert np.ma.is_masked(topo.heights()), "Masked array already lost before saving to netCDF"

    with tempfile.NamedTemporaryFile() as tmpfile:
        topo.to_netcdf(tmpfile.name)
        netcdf_topo = read_topography(tmpfile.name)
        assert np.ma.is_masked(netcdf_topo.heights()), "Masked array lost while saving to netCDF"


@pytest.mark.parametrize("dim, is_nonuniform, height_scale_factor, detrend_mode", [
    (2, False, None, None),  # None means: do not apply
    (2, False, 2, None),
    (2, False, None, 'center'),
    (2, False, 2, 'center'),
])
def test_exception(make_topography_with_missing_data, dim, is_nonuniform, height_scale_factor, detrend_mode):
    topo = make_topography_with_missing_data(dim=dim, is_nonuniform=is_nonuniform,
                                             height_scale_factor=height_scale_factor)

    assert topo.rms_height_from_profile() is not None
    assert topo.rms_height_from_area() is not None

    with pytest.raises(UndefinedDataError):
        topo.rms_slope_from_profile()

    with pytest.raises(UndefinedDataError):
        topo.rms_gradient()

    with pytest.raises(UndefinedDataError):
        topo.rms_curvature_from_profile()

    with pytest.raises(UndefinedDataError):
        topo.rms_curvature_from_area()

    with pytest.raises(UndefinedDataError):
        topo.rms_laplacian()

    with pytest.raises(UndefinedDataError):
        topo.autocorrelation_from_profile()

    with pytest.raises(UndefinedDataError):
        topo.autocorrelation_from_area()

    with pytest.raises(UndefinedDataError):
        topo.power_spectrum_from_profile()

    with pytest.raises(UndefinedDataError):
        topo.power_spectrum_from_area()
