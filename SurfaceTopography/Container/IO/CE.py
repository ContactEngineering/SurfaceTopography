#
# Copyright 2023-2024 Lars Pastewka
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
import warnings
from datetime import datetime
from zipfile import ZipFile

import numpy as np
import yaml

from ...DiscoverVersion import __version__
from ...IO import open_topography
from ..SurfaceContainer import LazySurfaceContainer, SurfaceContainer
from .Reader import ContainerReaderBase


class _ReadTopography(object):
    def __init__(self, reader, ignore_filters=False, **kwargs):
        self._reader = reader
        self._ignore_filters = ignore_filters
        self._kwargs = kwargs

    def __call__(self):
        # Read the topography from the preferred data file
        t = self._reader.topography(
            physical_sizes=self._kwargs['physical_sizes'],
            periodic=self._kwargs['periodic'],
            unit=self._kwargs['unit'],
            info=self._kwargs['info'],
            channel_index=self._kwargs['channel_index']
        )

        # We need to reconstruct the pipeline if the data file does
        # not contain squeezed data, currently indicate by a
        # 'squeezed' prefix to the data file key
        datafile_key = self._kwargs['datafile_key']
        topo_meta = self._kwargs['topo_meta']
        if not self._ignore_filters and not datafile_key.startswith('squeezed'):
            if 'height_scale' in topo_meta:
                t = t.scale(topo_meta['height_scale'])
            if 'fill_undefined_data_mode' in topo_meta and topo_meta['fill_undefined_data_mode'] != 'do-not-fill':
                t = t.interpolate_undefined_data(topo_meta['fill_undefined_data_mode'])
            if 'detrend_mode' in topo_meta:
                t = t.detrend(topo_meta['detrend_mode'])

        return t


class CEFileOpener(object):
    def __init__(self, zipname, filename):
        self._zipname = zipname
        self._filename = filename

    def __call__(self):
        return ZipFile(self._zipname, mode='r').open(self._filename, mode='r')


class CEReader(ContainerReaderBase):
    _format = 'ce'
    _mime_types = ['application/zip']
    _file_extensions = ['zip']

    _name = 'contact.engineering'
    _description = '''
    This reader imports digital surface twin containers from https://contact.engineering/.
    '''

    def __init__(self, fn, datafile_keys=['original', 'squeezed-netcdf'], ignore_filters=False):
        """
        Read all surfaces in a contact.engineering container file and associated
        metadata. The container is a ZIP file with raw data files and a YAML file
        meta.yml that contains all metadata not contained in the raw data files.

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
        ignore_filters : bool, optional
            If True, the filter pipeline is not (re-)constructed from the metadata.
            (Default: False)

        Returns
        -------
        surface_containers : list of :obj:`SurfaceContainer`s
            List of all surfaces contained in this container file.
        """
        self._fn = fn

        self._containers = []

        with ZipFile(self._fn, 'r') as z:
            meta = yaml.load(z.open('meta.yml'), Loader=yaml.FullLoader)

        readers = []
        for surf_meta in meta['surfaces']:
            for topo_meta in surf_meta['topographies']:
                info = topo_meta.copy()

                # Check whether the metadata contains sizes and unit. If not
                # this information comes from the data file.
                try:
                    physical_sizes = info['size']
                    del info['size']
                except KeyError:
                    physical_sizes = None
                try:
                    unit = info['unit']
                    del info['unit']
                except KeyError:
                    unit = None
                try:
                    periodic = info['is_periodic']
                    del info['is_periodic']
                except KeyError:
                    periodic = False
                datafile_key = None
                datafiles = topo_meta['datafile']

                # Pick first of the provided possible data file keys that
                # exists in the container
                for key in datafile_keys:
                    if key in datafiles:
                        datafile_key = key
                        break

                # There may be none, complain
                if datafile_key is None:
                    raise ValueError('Could not detect data file.')

                # Inspect topography file; we pass a function that returns a file handle to reopen file
                reader = open_topography(CEFileOpener(self._fn, datafiles[datafile_key]))

                # Channel to load
                if 'data_source' in topo_meta:
                    data_source = topo_meta['data_source']
                else:
                    data_source = reader.default_channel.index

                # Check consistency between data file and meta.yml
                physical_sizes_from_file = reader.channels[data_source].physical_sizes
                if physical_sizes_from_file is not None:
                    if physical_sizes is not None:
                        if not np.allclose(physical_sizes_from_file, physical_sizes, rtol=1e-4):
                            warnings.warn(f'Physical sizes from data file (={physical_sizes_from_file} and from '
                                          f'meta.yml (={physical_sizes}) differ for topography '
                                          f'{datafiles[datafile_key]}')
                        # Need to set this to None to avoid collision
                        physical_sizes = None

                unit_from_file = reader.channels[data_source].unit
                if unit_from_file is not None:
                    if unit is not None:
                        if unit_from_file != unit:
                            warnings.warn(f'Unit from data file (={unit_from_file}) and from meta.yml '
                                          f'(={unit}) differ for topography {datafiles[datafile_key]}')
                        # Need to set this to None to avoid collision
                        unit = None

                readers += [
                    _ReadTopography(reader,
                                    ignore_filters=ignore_filters,
                                    physical_sizes=physical_sizes,
                                    periodic=periodic,
                                    unit=unit,
                                    info=info,
                                    datafile_key=datafile_key,
                                    channel_index=data_source,
                                    topo_meta=topo_meta)
                ]

            self._containers += [LazySurfaceContainer(readers)]

    def container(self, index=0):
        """
        Returns an instance of a subclass of :obj:`SurfaceContainer` that
        contains a list of topographies.

        Arguments
        ---------
        index : int
            Index of the container to load.
            (Default: 0, which loads the first container)

        Returns
        -------
        surface_container : subclass of :obj:`SurfaceContainer`
            The object containing a list with actual topography data.
        """
        return self._containers[index]


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
