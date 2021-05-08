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
