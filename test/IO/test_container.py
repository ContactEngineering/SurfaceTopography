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

import tempfile

from numpy.testing import assert_allclose

from SurfaceTopography import SurfaceContainer, read_container, read_topography  # , read_published_container


def test_read_just_uniform(file_format_examples):
    for c, in [
        read_container(f'{file_format_examples}/container1.zip'),
        # read_published_container('https://contact.engineering/go/867nv/')  # Same a container1.zip
        # TODO maybe this makes the web app stall when running MPI tests von Travis, further investigation needed
    ]:
        assert len(c) == 3

        assert c[0].nb_grid_pts == (500, 500)
        assert c[1].nb_grid_pts == (500, 500)
        assert c[2].nb_grid_pts == (500, 500)

        assert c[0].physical_sizes == (100, 100)
        assert c[1].physical_sizes == (10, 10)
        assert c[2].physical_sizes == (1, 1)

        assert c[0].info['unit'] == 'µm'
        assert c[1].info['unit'] == 'µm'
        assert c[2].info['unit'] == 'µm'


def test_read_mixed(file_format_examples):
    c, = read_container(f'{file_format_examples}/container2.zip')

    assert len(c) == 3


def test_write(file_format_examples):
    t1 = read_topography(f'{file_format_examples}/di1.di')
    t2 = read_topography(f'{file_format_examples}/example.opd')
    t3 = read_topography(f'{file_format_examples}/example2.txt')

    c = SurfaceContainer([t1, t2, t3])

    with tempfile.TemporaryFile() as fobj:
        c.to_zip(fobj)

        c2, = read_container(fobj)

        assert len(c2) == 3

        assert c2[0].nb_grid_pts == t1.nb_grid_pts
        assert c2[1].nb_grid_pts == t2.nb_grid_pts
        assert c2[2].nb_grid_pts == t3.nb_grid_pts

        assert_allclose(c2[0].physical_sizes, t1.physical_sizes)
        assert_allclose(c2[1].physical_sizes, t2.physical_sizes)
        assert_allclose(c2[2].physical_sizes, t3.physical_sizes)

        assert c2[0].info['unit'] == t1.info['unit']
        assert c2[1].info['unit'] == t2.info['unit']
        assert c2[2].info['unit'] == t3.info['unit']
