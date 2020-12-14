import os
import unittest

from SurfaceTopography.IO.FromFile import read_opd

DATADIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))),
    'file_format_examples')


class opdSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_read(self):
        surface = read_opd(os.path.join(DATADIR, 'example.opd'))
        nx, ny = surface.nb_grid_pts
        self.assertEqual(nx, 640)
        self.assertEqual(ny, 480)
        sx, sy = surface.physical_sizes
        self.assertAlmostEqual(sx, 0.125909140)
        self.assertAlmostEqual(sy, 0.094431855)
        self.assertTrue(surface.is_uniform)

    def test_undefined_points(self):
        t = read_opd(os.path.join(DATADIR, 'example2.opd'))
        self.assertTrue(t.has_undefined_data)
