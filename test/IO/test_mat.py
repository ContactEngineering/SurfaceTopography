import os
import numpy as np

from SurfaceTopography.IO import MatReader

DATADIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))),
    'file_format_examples')


def test_read():
    surface = MatReader(os.path.join(DATADIR, 'example1.mat')).topography(
        physical_sizes=[1., 1.])
    nx, ny = surface.nb_grid_pts
    assert nx == 2048
    assert ny == 2048
    np.testing.assert_almost_equal(surface.rms_height(), 1.234061e-07)
    assert surface.is_uniform

# TODO: test with multiple data
