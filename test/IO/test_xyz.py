import os

from SurfaceTopography.IO.FromFile import read_xyz

DATADIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))),
    'file_format_examples')


def test_read():
    surface = read_xyz(os.path.join(DATADIR, 'example.asc'))
    assert not surface.is_uniform
    x, y = surface.positions_and_heights()
    assert len(x) > 0
    assert len(x) == len(y)
    assert not surface.is_uniform
    assert surface.dim == 1
    assert not surface.is_periodic
