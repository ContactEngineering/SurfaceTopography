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
import textwrap
import yaml
from datetime import datetime
from zipfile import ZipFile

try:
    from importlib.metadata import version

    __version__ = version('SurfaceTopography')
except ImportError:
    from pkg_resources import get_distribution

    __version__ = get_distribution('SurfaceTopography').version

from ..IO import read_topography
from .SurfaceContainer import SurfaceContainer


def read_container(fn, datafile_keys=['original', 'squeezed-netcdf']):
    """
    Read all surface in a container file and associated metadata. The
    container is a ZIP file with raw data files and a YAML file meta.yml
    that contains all metadata not contained in the raw data files.

    Parameters
    ----------
    fn : str or stream
        File or stream that contains the ZIP-container.
    datafile_keys : list of str, optional
        List of possible keys in 'meta.yml' that contains the name of the
        datafile to open. Code will try these keys in order. If a key
        starts with 'squeezed', the pipeline is not constructed from
        the metadata.
        (Default: ['original', 'squeezed-netcdf'])

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
                info = topo_meta.copy()
                physical_sizes = None
                if 'size' in info:
                    physical_sizes = info['size']
                    del info['size']
                datafile_key = None
                datafiles = topo_meta['datafile']
                for key in datafiles:
                    datafile_key = key
                if datafile_key is None:
                    raise ValueError('Could not detect data file.')
                t = read_topography(
                    z.open(datafiles[datafile_key]),
                    physical_sizes=physical_sizes,
                    info=info
                )
                if not datafile_key.startswith('squeezed'):
                    if 'height_scale' in topo_meta:
                        t = t.scale(topo_meta['height_scale'])
                    if 'detrend_mode' in topo_meta:
                        t = t.detrend(topo_meta['detrend_mode'])
                topographies += [t]

            surfaces += [SurfaceContainer(topographies)]

    return surfaces


def write_containers(containers, fn):
    """
    Write multiple surface containers into a ZIP file.

    Parameters
    ----------
    containers : list of :obj:`SurfaceContainer`s
        Containers to be written to the ZIP file.
    fn : str or stream
        File or stream to write the ZIP-container to.
    """

    # This is adapted from TopoBank

    surfaces_dicts = []
    counter = 0

    with ZipFile(fn, mode='w') as zf:
        #
        # Add meta data and topography files for all given surfaces
        #
        for surface_container in containers:
            topography_dicts = []

            for topography in surface_container:
                # Create unique file names for the data files by simply appending a counter
                topofile_name = f'topography{counter}.nc'
                counter += 1

                # We do not write the info dictionary into the YAML metadata as it is already container in the NetCDF
                # file (serialized as JSON)
                topo_meta = {
                    'datafile': {
                        'squeezed-netcdf': topofile_name
                    }
                }

                # Add topography file as NetCDF to the ZIP archive
                with tempfile.TemporaryFile() as f:
                    topography.to_netcdf(f)
                    f.seek(0)
                    zf.writestr(topofile_name, f.read())

                topography_dicts.append(topo_meta)

            surface_dict = {'topographies': topography_dicts}
            surfaces_dicts.append(surface_dict)

        #
        # Add metadata file
        #
        metadata = {
            'versions': {'SurfaceTopography': __version__},
            'surfaces': surfaces_dicts,
            'creation_time': str(datetime.now()),
        }

        zf.writestr("meta.yml", yaml.dump(metadata))

        #
        # Add a Readme file
        #
        readme_txt = textwrap.dedent(f"""
        Contents of this ZIP archive
        ============================
        This archive contains {len(containers)} surface(s). Each surface is a
        collection of individual topography measurements.
        In total {sum(len(x) for x in containers)} topography measurements are included.

        The meta data for the surfaces and the individual topographies
        can be found in the auxiliary file 'meta.yml' and within
        the NetCDF topographies themselves. 'meta.yml' is formatted
        as a [YAML](https://yaml.org/) file. The NetCDF topographies
        contain metadata as [JSON](https://www.json.org/).

        Version information
        ===================

        SurfaceTopography: {__version__}
        """)

        zf.writestr("README.txt", textwrap.dedent(readme_txt))


SurfaceContainer.register_function('to_zip', lambda container, fn: write_containers([container], fn))
