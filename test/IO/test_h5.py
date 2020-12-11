import os
import unittest

import SurfaceTopography
from SurfaceTopography.IO import H5Reader

DATADIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))),
    'file_format_examples')


class h5SurfaceTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_detect_format(self):
        self.assertEqual(SurfaceTopography.IO.detect_format(
            # TODO(pastewka): this will be the standard detect format method
            #  in the future
            os.path.join(DATADIR, 'surface.2048x2048.h5')), 'h5')

    def test_read(self):
        loader = H5Reader(os.path.join(DATADIR, 'surface.2048x2048.h5'))

        topography = loader.topography(physical_sizes=(1., 1.))
        nx, ny = topography.nb_grid_pts
        self.assertEqual(nx, 2048)
        self.assertEqual(ny, 2048)
        self.assertTrue(topography.is_uniform)
        self.assertEqual(topography.dim, 2)
