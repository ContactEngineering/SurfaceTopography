import pytest
import os.path
import tempfile
import numpy as np

from SurfaceTopography.IO import open_topography, read_topography


@pytest.fixture
def make_topography_with_missing_data(file_format_examples):
    """Returns a topography with missing data, optionally scaled.

    Returns
    -------
    Factory function for a topography with missing data.
    """

    def _make_topo(height_scale_factor = None):
        """
        Parameters
        ----------

        height_scale_factor: float or None
            If given, the topography will be scaled by this factor

        Returns
        -------
        Topography with missing data.
        """
        r = open_topography(os.path.join(file_format_examples, 'missing.xyz'))
        topo_kwargs = {}
        if height_scale_factor is not None:
            topo_kwargs['height_scale_factor'] = height_scale_factor
        return r.topography(**topo_kwargs)

    return _make_topo


@pytest.mark.parametrize("height_scale_factor, detrend_mode", [
    (None, None),  # None means: do not apply
    (1, None),
    (None, 'center'),
    (1, 'center')])
def test_masked_array_kept_by_netcdf(make_topography_with_missing_data, height_scale_factor, detrend_mode):
    """For different ways in the pipeline .to_netcdf() should keep masked data."""
    topo = make_topography_with_missing_data(height_scale_factor=height_scale_factor)
    if detrend_mode is not None:
        topo = topo.detrend(detrend_mode)

    assert np.ma.is_masked(topo.heights()), "Masked array already lost before saving to netCDF"

    with tempfile.NamedTemporaryFile() as tmpfile:
        topo.to_netcdf(tmpfile.name)
        netcdf_topo = read_topography(tmpfile.name)
        assert np.ma.is_masked(netcdf_topo.heights()), "Masked array lost while saving to netCDF"
