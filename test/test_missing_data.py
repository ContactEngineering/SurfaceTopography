import os
import tempfile

import numpy as np
import pytest

from NuMPI import MPI

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
            fn = 'missing_data_2d.xyz'
        elif dim == 1:
            if is_nonuniform:
                fn = 'missing_data_1d_nonuniform.txt'
            else:
                fn = 'missing_data_1d_uniform.txt'

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
