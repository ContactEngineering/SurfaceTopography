#
# Copyright 2020-2021, 2023 Lars Pastewka
#           2019-2020 Antoine Sanner
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

import json
import os
import tempfile
import xml.etree.cElementTree as ET

import numpy as np
import pytest

from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.Generation import fourier_synthesis

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_write_xml():
    nx, ny = 1782, 1302
    t = fourier_synthesis((nx, ny), (1, 1), 0.8, rms_slope=0.1, unit='mm')
    sx, sy = t.physical_sizes
    with tempfile.TemporaryDirectory() as d:
        manifest = t.to_dzi('synthetic', d)
        assert os.path.exists(f'{d}/synthetic_files')
        for i in range(12):
            assert os.path.exists(f'{d}/synthetic_files/{i}')
        root = ET.parse(open(f'{d}/synthetic.xml')).getroot()
        assert root.attrib['TileSize'] == '256'
        assert root.attrib['Overlap'] == '1'
        assert root.attrib['Format'] == 'jpg'
        assert root.attrib['ColorbarTitle'] == 'Height (µm)'  # The writer decided to use µm, not mm
        assert root.attrib['Colormap'] == 'viridis'
        assert root[0].attrib['Width'] == f'{nx}'
        assert root[0].attrib['Height'] == f'{ny}'
        np.testing.assert_allclose(float(root[1].attrib['Width']), 1000 * nx / sx)
        np.testing.assert_allclose(float(root[1].attrib['Height']), 1000 * ny / sy)
        np.testing.assert_allclose(float(root[2].attrib['Minimum']), 1000 * t.min())
        np.testing.assert_allclose(float(root[2].attrib['Maximum']), 1000 * t.max())

        manifest = [fn[len(d) + 1:] for fn in manifest]
        assert set(manifest) == set([
            'synthetic.xml',
            'synthetic_files/11/0_0.jpg', 'synthetic_files/11/0_1.jpg', 'synthetic_files/11/0_2.jpg',
            'synthetic_files/11/0_3.jpg', 'synthetic_files/11/0_4.jpg', 'synthetic_files/11/0_5.jpg',
            'synthetic_files/11/1_0.jpg', 'synthetic_files/11/1_1.jpg', 'synthetic_files/11/1_2.jpg',
            'synthetic_files/11/1_3.jpg', 'synthetic_files/11/1_4.jpg', 'synthetic_files/11/1_5.jpg',
            'synthetic_files/11/2_0.jpg', 'synthetic_files/11/2_1.jpg', 'synthetic_files/11/2_2.jpg',
            'synthetic_files/11/2_3.jpg', 'synthetic_files/11/2_4.jpg', 'synthetic_files/11/2_5.jpg',
            'synthetic_files/11/3_0.jpg', 'synthetic_files/11/3_1.jpg', 'synthetic_files/11/3_2.jpg',
            'synthetic_files/11/3_3.jpg', 'synthetic_files/11/3_4.jpg',
            'synthetic_files/11/3_5.jpg', 'synthetic_files/11/4_0.jpg', 'synthetic_files/11/4_1.jpg',
            'synthetic_files/11/4_2.jpg', 'synthetic_files/11/4_3.jpg', 'synthetic_files/11/4_4.jpg',
            'synthetic_files/11/4_5.jpg', 'synthetic_files/11/5_0.jpg', 'synthetic_files/11/5_1.jpg',
            'synthetic_files/11/5_2.jpg', 'synthetic_files/11/5_3.jpg', 'synthetic_files/11/5_4.jpg',
            'synthetic_files/11/5_5.jpg', 'synthetic_files/11/6_0.jpg', 'synthetic_files/11/6_1.jpg',
            'synthetic_files/11/6_2.jpg', 'synthetic_files/11/6_3.jpg', 'synthetic_files/11/6_4.jpg',
            'synthetic_files/11/6_5.jpg', 'synthetic_files/10/0_0.jpg', 'synthetic_files/10/0_1.jpg',
            'synthetic_files/10/0_2.jpg', 'synthetic_files/10/1_0.jpg', 'synthetic_files/10/1_1.jpg',
            'synthetic_files/10/1_2.jpg', 'synthetic_files/10/2_0.jpg', 'synthetic_files/10/2_1.jpg',
            'synthetic_files/10/2_2.jpg', 'synthetic_files/10/3_0.jpg', 'synthetic_files/10/3_1.jpg',
            'synthetic_files/10/3_2.jpg', 'synthetic_files/9/0_0.jpg', 'synthetic_files/9/0_1.jpg',
            'synthetic_files/9/1_0.jpg', 'synthetic_files/9/1_1.jpg', 'synthetic_files/8/0_0.jpg',
            'synthetic_files/7/0_0.jpg', 'synthetic_files/6/0_0.jpg', 'synthetic_files/5/0_0.jpg',
            'synthetic_files/4/0_0.jpg', 'synthetic_files/3/0_0.jpg', 'synthetic_files/2/0_0.jpg',
            'synthetic_files/1/0_0.jpg', 'synthetic_files/0/0_0.jpg'])


def test_write_json():
    nx, ny = 1324, 871
    t = fourier_synthesis((nx, ny), (1.3, 1.2), 0.8, rms_slope=0.1, unit='mm')
    sx, sy = t.physical_sizes
    with tempfile.TemporaryDirectory() as d:
        manifest = t.to_dzi('synthetic', d, meta_format='json')
        assert os.path.exists(f'{d}/synthetic_files')
        for i in range(12):
            assert os.path.exists(f'{d}/synthetic_files/{i}')
        with open(f'{d}/synthetic.json', 'r') as f:
            meta = json.load(f)
        meta = meta['Image']
        assert meta['TileSize'] == 256
        assert meta['Overlap'] == 1
        assert meta['Format'] == 'jpg'
        assert meta['ColorbarTitle'] == 'Height (µm)'  # The writer decided to use µm, not mm
        assert meta['Colormap'] == 'viridis'
        assert meta['Size']['Width'] == nx
        assert meta['Size']['Height'] == ny
        np.testing.assert_allclose(meta['PixelsPerMeter']['Width'], 1000 * nx / sx)
        np.testing.assert_allclose(meta['PixelsPerMeter']['Height'], 1000 * ny / sy)
        np.testing.assert_allclose(meta['ColorbarRange']['Minimum'], t.min() * 1000)
        np.testing.assert_allclose(meta['ColorbarRange']['Maximum'], t.max() * 1000)

        manifest = [fn[len(d) + 1:] for fn in manifest]
        assert set(manifest) == set([
            'synthetic.json',
            'synthetic_files/0/0_0.jpg', 'synthetic_files/1/0_0.jpg', 'synthetic_files/10/0_0.jpg',
            'synthetic_files/10/0_1.jpg', 'synthetic_files/10/1_0.jpg', 'synthetic_files/10/1_1.jpg',
            'synthetic_files/10/2_0.jpg', 'synthetic_files/10/2_1.jpg', 'synthetic_files/11/0_0.jpg',
            'synthetic_files/11/0_1.jpg', 'synthetic_files/11/0_2.jpg', 'synthetic_files/11/0_3.jpg',
            'synthetic_files/11/1_0.jpg', 'synthetic_files/11/1_1.jpg', 'synthetic_files/11/1_2.jpg',
            'synthetic_files/11/1_3.jpg', 'synthetic_files/11/2_0.jpg', 'synthetic_files/11/2_1.jpg',
            'synthetic_files/11/2_2.jpg', 'synthetic_files/11/2_3.jpg', 'synthetic_files/11/3_0.jpg',
            'synthetic_files/11/3_1.jpg', 'synthetic_files/11/3_2.jpg', 'synthetic_files/11/3_3.jpg',
            'synthetic_files/11/4_0.jpg', 'synthetic_files/11/4_1.jpg', 'synthetic_files/11/4_2.jpg',
            'synthetic_files/11/4_3.jpg', 'synthetic_files/11/5_0.jpg', 'synthetic_files/11/5_1.jpg',
            'synthetic_files/11/5_2.jpg', 'synthetic_files/11/5_3.jpg', 'synthetic_files/2/0_0.jpg',
            'synthetic_files/3/0_0.jpg', 'synthetic_files/4/0_0.jpg', 'synthetic_files/5/0_0.jpg',
            'synthetic_files/6/0_0.jpg', 'synthetic_files/7/0_0.jpg', 'synthetic_files/8/0_0.jpg',
            'synthetic_files/9/0_0.jpg', 'synthetic_files/9/1_0.jpg'])


def test_write_netcdf():
    nx, ny = 1324, 871
    t = fourier_synthesis((nx, ny), (1.3, 1.2), 0.8, rms_slope=0.1, unit='mm')
    sx, sy = t.physical_sizes
    with tempfile.TemporaryDirectory() as d:
        manifest = t.to_dzi('synthetic', d, format='nc', meta_format='json')
        assert os.path.exists(f'{d}/synthetic_files')
        for i in range(12):
            assert os.path.exists(f'{d}/synthetic_files/{i}')
        with open(f'{d}/synthetic.json', 'r') as f:
            meta = json.load(f)
        meta = meta['Image']
        assert meta['TileSize'] == 256
        assert meta['Overlap'] == 1
        assert meta['Format'] == 'nc'
        assert meta['ColorbarTitle'] == 'Height (µm)'  # The writer decided to use µm, not mm
        assert meta['Colormap'] == 'viridis'
        assert meta['Size']['Width'] == nx
        assert meta['Size']['Height'] == ny
        np.testing.assert_allclose(meta['PixelsPerMeter']['Width'], 1000 * nx / sx)
        np.testing.assert_allclose(meta['PixelsPerMeter']['Height'], 1000 * ny / sy)
        np.testing.assert_allclose(meta['ColorbarRange']['Minimum'], t.min() * 1000)
        np.testing.assert_allclose(meta['ColorbarRange']['Maximum'], t.max() * 1000)

        manifest = [fn[len(d) + 1:] for fn in manifest]
        assert set(manifest) == set([
            'synthetic.json',
            'synthetic_files/0/0_0.nc', 'synthetic_files/1/0_0.nc', 'synthetic_files/10/0_0.nc',
            'synthetic_files/10/0_1.nc', 'synthetic_files/10/1_0.nc', 'synthetic_files/10/1_1.nc',
            'synthetic_files/10/2_0.nc', 'synthetic_files/10/2_1.nc', 'synthetic_files/11/0_0.nc',
            'synthetic_files/11/0_1.nc', 'synthetic_files/11/0_2.nc', 'synthetic_files/11/0_3.nc',
            'synthetic_files/11/1_0.nc', 'synthetic_files/11/1_1.nc', 'synthetic_files/11/1_2.nc',
            'synthetic_files/11/1_3.nc', 'synthetic_files/11/2_0.nc', 'synthetic_files/11/2_1.nc',
            'synthetic_files/11/2_2.nc', 'synthetic_files/11/2_3.nc', 'synthetic_files/11/3_0.nc',
            'synthetic_files/11/3_1.nc', 'synthetic_files/11/3_2.nc', 'synthetic_files/11/3_3.nc',
            'synthetic_files/11/4_0.nc', 'synthetic_files/11/4_1.nc', 'synthetic_files/11/4_2.nc',
            'synthetic_files/11/4_3.nc', 'synthetic_files/11/5_0.nc', 'synthetic_files/11/5_1.nc',
            'synthetic_files/11/5_2.nc', 'synthetic_files/11/5_3.nc', 'synthetic_files/2/0_0.nc',
            'synthetic_files/3/0_0.nc', 'synthetic_files/4/0_0.nc', 'synthetic_files/5/0_0.nc',
            'synthetic_files/6/0_0.nc', 'synthetic_files/7/0_0.nc', 'synthetic_files/8/0_0.nc',
            'synthetic_files/9/0_0.nc', 'synthetic_files/9/1_0.nc'])

        # Try reading one of the files
        t = read_topography(f'{d}/synthetic_files/11/4_1.nc')
        assert t.nb_grid_pts == (258, 258)
        np.testing.assert_allclose(t.physical_sizes, (0.981873, 1.377727), rtol=1e-6)


def test_write_npy():
    nx, ny = 1324, 871
    t = fourier_synthesis((nx, ny), (1.3, 1.2), 0.8, rms_slope=0.1, unit='mm')
    sx, sy = t.physical_sizes
    with tempfile.TemporaryDirectory() as d:
        manifest = t.to_dzi('synthetic', d, format='npy', meta_format='json')
        assert os.path.exists(f'{d}/synthetic_files')
        for i in range(12):
            assert os.path.exists(f'{d}/synthetic_files/{i}')
        with open(f'{d}/synthetic.json', 'r') as f:
            meta = json.load(f)
        meta = meta['Image']
        assert meta['TileSize'] == 256
        assert meta['Overlap'] == 1
        assert meta['Format'] == 'npy'
        assert meta['ColorbarTitle'] == 'Height (µm)'  # The writer decided to use µm, not mm
        assert meta['Colormap'] == 'viridis'
        assert meta['Size']['Width'] == nx
        assert meta['Size']['Height'] == ny
        np.testing.assert_allclose(meta['PixelsPerMeter']['Width'], 1000 * nx / sx)
        np.testing.assert_allclose(meta['PixelsPerMeter']['Height'], 1000 * ny / sy)
        np.testing.assert_allclose(meta['ColorbarRange']['Minimum'], t.min() * 1000)
        np.testing.assert_allclose(meta['ColorbarRange']['Maximum'], t.max() * 1000)

        manifest = [fn[len(d) + 1:] for fn in manifest]
        assert set(manifest) == set([
            'synthetic.json',
            'synthetic_files/0/0_0.npy', 'synthetic_files/1/0_0.npy', 'synthetic_files/10/0_0.npy',
            'synthetic_files/10/0_1.npy', 'synthetic_files/10/1_0.npy', 'synthetic_files/10/1_1.npy',
            'synthetic_files/10/2_0.npy', 'synthetic_files/10/2_1.npy', 'synthetic_files/11/0_0.npy',
            'synthetic_files/11/0_1.npy', 'synthetic_files/11/0_2.npy', 'synthetic_files/11/0_3.npy',
            'synthetic_files/11/1_0.npy', 'synthetic_files/11/1_1.npy', 'synthetic_files/11/1_2.npy',
            'synthetic_files/11/1_3.npy', 'synthetic_files/11/2_0.npy', 'synthetic_files/11/2_1.npy',
            'synthetic_files/11/2_2.npy', 'synthetic_files/11/2_3.npy', 'synthetic_files/11/3_0.npy',
            'synthetic_files/11/3_1.npy', 'synthetic_files/11/3_2.npy', 'synthetic_files/11/3_3.npy',
            'synthetic_files/11/4_0.npy', 'synthetic_files/11/4_1.npy', 'synthetic_files/11/4_2.npy',
            'synthetic_files/11/4_3.npy', 'synthetic_files/11/5_0.npy', 'synthetic_files/11/5_1.npy',
            'synthetic_files/11/5_2.npy', 'synthetic_files/11/5_3.npy', 'synthetic_files/2/0_0.npy',
            'synthetic_files/3/0_0.npy', 'synthetic_files/4/0_0.npy', 'synthetic_files/5/0_0.npy',
            'synthetic_files/6/0_0.npy', 'synthetic_files/7/0_0.npy', 'synthetic_files/8/0_0.npy',
            'synthetic_files/9/0_0.npy', 'synthetic_files/9/1_0.npy'])
