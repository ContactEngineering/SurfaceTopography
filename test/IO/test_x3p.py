#
# Copyright 2019-2020 Lars Pastewka
#           2019 Michael Röttger
#           2019 Kai Haase
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
