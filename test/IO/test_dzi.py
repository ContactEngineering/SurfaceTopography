#
# Copyright 2021 Lars Pastewka
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
import tempfile
import xml.etree.cElementTree as ET

from SurfaceTopography.Generation import fourier_synthesis


def test_write():
    nx, ny = 1782, 1302
    t = fourier_synthesis((nx, ny), (1, 1), 0.8, rms_slope=0.1)
    with tempfile.TemporaryDirectory() as d:
        filenames = t.to_dzi('synthetic', d)
        assert os.path.exists(f'{d}/synthetic_files')
        for i in range(12):
            assert os.path.exists(f'{d}/synthetic_files/{i}')
        root = ET.parse(open(f'{d}/synthetic.xml')).getroot()
        assert root.attrib['TileSize'] == '256'
        assert root.attrib['Overlap'] == '1'
        assert root.attrib['Format'] == 'jpg'
        assert root[0].attrib['Width'] == f'{nx}'
        assert root[0].attrib['Height'] == f'{ny}'

        filenames = [fn[len(d)+1:] for fn in filenames]
        assert set(filenames) == set([
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
