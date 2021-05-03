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

import yaml
from zipfile import ZipFile

from ..IO import read_topography
from .SurfaceContainer import SurfaceContainer


def read_container(fn, datafile_key='datafile'):
    """
    Read all surface in a container file and associated metadata. The
    container is a ZIP file with raw data files and a YAML file meta.yml
    that contains all metadata not contained in the raw data files.

    Parameters
    ----------
    fn : str or stream
        File or stream that contains the ZIP-container
    datafile_key : str, optional
        Key in 'meta.yml' that contains the name of the datafile to open.
        (Default: 'datafile')

    Returns
    -------
    surface_containers : list of :obj:`SurfaceContainer`s
        List of all surfaces contained in this container file.
    """

    surfaces = []

    with ZipFile(fn, 'r') as z:
        meta = yaml.load(z.open('meta.yml'))

        topographies = []
        for surf_meta in meta['surfaces']:
            for topo_meta in surf_meta['topographies']:
                t = read_topography(
                    z.open(topo_meta[datafile_key]),
                    physical_sizes=topo_meta['size'],
                    info=dict(
                        unit=topo_meta['unit'],
                        measurement_date=topo_meta['measurement_date'],
                        description=topo_meta['description'],
                        creator=topo_meta['creator']
                    )
                ).scale(topo_meta['height_scale']).detrend(topo_meta['detrend_mode'])
                topographies += [t]

            surfaces += [SurfaceContainer(topographies)]

    return surfaces
