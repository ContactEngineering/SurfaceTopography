import os
import unittest

import numpy as np

from SurfaceTopography.IO.FromFile import read_x3p

DATADIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))),
    'file_format_examples')


class x3pSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_read(self):
        surface = read_x3p(os.path.join(DATADIR, 'example.x3p'))
        nx, ny = surface.nb_grid_pts
        self.assertEqual(nx, 777)
        self.assertEqual(ny, 1035)
        sx, sy = surface.physical_sizes
        self.assertAlmostEqual(sx, 0.00068724)
        self.assertAlmostEqual(sy, 0.00051593)
        surface = read_x3p(os.path.join(DATADIR, 'example2.x3p'))
        nx, ny = surface.nb_grid_pts
        self.assertEqual(nx, 650)
        self.assertEqual(ny, 650)
        sx, sy = surface.physical_sizes
        self.assertAlmostEqual(sx, 8.29767313942749e-05)
        self.assertAlmostEqual(sy, 0.0002044783737930349)
        self.assertTrue(surface.is_uniform)

    def test_points_for_uniform_topography(self):
        surface = read_x3p(os.path.join(DATADIR, 'example.x3p'))
        x, y, z = surface.positions_and_heights()
        self.assertAlmostEqual(np.mean(np.diff(x[:, 0])),
                               surface.physical_sizes[0] / surface.nb_grid_pts[
                                   0])
        self.assertAlmostEqual(np.mean(np.diff(y[0, :])),
                               surface.physical_sizes[1] / surface.nb_grid_pts[
                                   1])
