import os

from SurfaceTopography.IO.FromFile import read_hgt

DATADIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))),
    'file_format_examples')


def test_read():
    surface = read_hgt(os.path.join(DATADIR, 'N46E013.hgt'))
    nx, ny = surface.nb_grid_pts
    assert nx == 3601
    assert ny == 3601
    assert surface.is_uniform
