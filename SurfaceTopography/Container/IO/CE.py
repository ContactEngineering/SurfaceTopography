#
# Copyright 2023-2026 Lars Pastewka
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
import textwrap
import warnings
from datetime import datetime
from zipfile import ZipFile

import numpy as np
import yaml
from pydantic import ValidationError

from ...Exceptions import CorruptFile, FileFormatMismatch
from ...Version import __version__
from ..SurfaceContainer import LazySurfaceContainer, SurfaceContainer
from .Reader import ContainerMember, ContainerReaderBase
from .Schema import (
    CONTAINER_METADATA_FILENAME,
    LEGACY_METADATA_FILENAME,
    ContainerMeta,
    DatafileMeta,
    SurfaceMeta,
    TopographyMeta,
)


class CEFileOpener(object):
    def __init__(self, zipname, filename):
        self._zipname = zipname
        self._filename = filename

    def __call__(self):
        zipfile = ZipFile(self._zipname, mode="r")
        stream = zipfile.open(self._filename, mode="r")
        # Close the containing ZipFile together with the member stream;
        # otherwise its file descriptor leaks until garbage collection
        original_close = stream.close

        def close():
            original_close()
            zipfile.close()

        stream.close = close
        return stream


class _ReadTopography(object):
    """
    Callable that opens and reads a single measurement of a CE container.

    The data file is opened (and its consistency with the container metadata
    checked) on first call, not when the container is opened. This keeps
    opening a container cheap — required for format detection, which
    trial-opens readers — and means a corrupt data file surfaces when the
    topography is accessed, not when the archive is opened.
    """

    def __init__(self, opener, datafile_key, topo_meta, ignore_filters=False):
        self._opener = opener
        self._datafile_key = datafile_key
        self._topo_meta = topo_meta
        self._ignore_filters = ignore_filters

    def __call__(self):
        # Lazy import to avoid circular dependency during package initialization
        from ...IO import open_topography

        topo_meta = self._topo_meta
        info = topo_meta.copy()

        # Check whether the metadata contains sizes and unit. If not
        # this information comes from the data file.
        try:
            physical_sizes = info["size"]
            del info["size"]
        except KeyError:
            physical_sizes = None
        try:
            unit = info["unit"]
            del info["unit"]
        except KeyError:
            unit = None
        try:
            periodic = info["is_periodic"]
            del info["is_periodic"]
        except KeyError:
            periodic = False

        # Inspect topography file; we pass a function that returns a file
        # handle to reopen the file
        reader = open_topography(self._opener)

        # Channel to load
        if "data_source" in topo_meta:
            data_source = topo_meta["data_source"]
        else:
            data_source = reader.default_channel.index

        # Check consistency between data file and container metadata
        physical_sizes_from_file = reader.channels[data_source].physical_sizes
        if physical_sizes_from_file is not None:
            if physical_sizes is not None:
                if not np.allclose(
                    physical_sizes_from_file, physical_sizes, rtol=1e-4
                ):
                    warnings.warn(
                        f"Physical sizes from data file (={physical_sizes_from_file} and from "
                        f"container metadata (={physical_sizes}) differ for topography "
                        f"{self._datafile_key}"
                    )
                # Need to set this to None to avoid collision
                physical_sizes = None

        unit_from_file = reader.channels[data_source].unit
        if unit_from_file is not None:
            if unit is not None:
                if unit_from_file != unit:
                    warnings.warn(
                        f"Unit from data file (={unit_from_file}) and from container "
                        f"metadata (={unit}) differ for topography {self._datafile_key}"
                    )
                # Need to set this to None to avoid collision
                unit = None

        # Read the topography from the preferred data file
        t = reader.topography(
            physical_sizes=physical_sizes,
            periodic=periodic,
            unit=unit,
            info=info,
            channel_index=data_source,
        )

        # We need to reconstruct the pipeline if the data file does
        # not contain squeezed data, currently indicate by a
        # 'squeezed' prefix to the data file key
        if not self._ignore_filters and not self._datafile_key.startswith("squeezed"):
            if "height_scale" in topo_meta:
                t = t.scale(topo_meta["height_scale"])
            if (
                "fill_undefined_data_mode" in topo_meta
                and topo_meta["fill_undefined_data_mode"] != "do-not-fill"
            ):
                t = t.interpolate_undefined_data(topo_meta["fill_undefined_data_mode"])
            if "detrend_mode" in topo_meta:
                t = t.detrend(topo_meta["detrend_mode"])

        return t


class CEReader(ContainerReaderBase):
    _format = "ce"
    _mime_types = ["application/zip"]
    _file_extensions = ["zip"]

    _name = "contact.engineering"
    _description = """
    This reader imports digital surface twin containers from https://contact.engineering/.
    """

    def __init__(
        self, fn, datafile_keys=["original", "squeezed-netcdf"], ignore_filters=False
    ):
        """
        Read all surfaces in a contact.engineering container file and associated
        metadata. The container is a ZIP file with raw data files and a single
        metadata file: canonically `index.json` (validated against
        :class:`SurfaceTopography.Container.IO.Schema.ContainerMeta`), with a
        legacy `meta.yml` accepted as fallback for archives written by older
        versions of TopoBank or SurfaceTopography.

        Opening a container parses only the metadata; the data files themselves
        are opened when the individual topographies are accessed.

        Parameters
        ----------
        fn : str or stream
            File or stream that contains the ZIP-container.
        datafile_keys : list of str, optional
            List of possible keys in the metadata that contains the name of the
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
        self._members = []

        with ZipFile(self._fn, "r") as z:
            names = set(z.namelist())
            if CONTAINER_METADATA_FILENAME in names:
                # Canonical metadata: strictly validated `index.json`. We
                # convert the validated model back into the dictionary shape
                # of the legacy YAML metadata, so that both paths below are
                # identical.
                with z.open(CONTAINER_METADATA_FILENAME, mode="r") as f:
                    try:
                        meta = ContainerMeta.model_validate_json(
                            f.read()
                        ).model_dump(by_alias=True, exclude_none=True)
                    except ValidationError as exc:
                        raise CorruptFile(
                            f"The container's '{CONTAINER_METADATA_FILENAME}' does not "
                            f"conform to the container metadata schema: {exc}"
                        ) from exc
            elif LEGACY_METADATA_FILENAME in names:
                # Legacy metadata: lenient. Old archives (including those
                # written by older versions of this library) may lack fields
                # that the schema requires, such as measurement names or
                # sizes; they are still read on a best-effort basis. The full
                # loader is required because old contact.engineering archives
                # contain YAML-serialized Python tuples.
                with z.open(LEGACY_METADATA_FILENAME, mode="r") as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)
            else:
                raise FileFormatMismatch(
                    f"This is not a contact.engineering container: it contains "
                    f"neither '{CONTAINER_METADATA_FILENAME}' nor "
                    f"'{LEGACY_METADATA_FILENAME}'."
                )

        for surf_meta in meta["surfaces"]:
            readers = []
            members = []
            for i, topo_meta in enumerate(surf_meta["topographies"]):
                datafiles = topo_meta["datafile"]

                # Pick first of the provided possible data file keys that
                # exists in the container
                datafile_key = None
                for key in datafile_keys:
                    if datafiles.get(key) is not None:
                        datafile_key = key
                        break

                # There may be none, complain
                if datafile_key is None:
                    raise CorruptFile("Could not detect data file.")

                opener = CEFileOpener(self._fn, datafiles[datafile_key])

                readers += [
                    _ReadTopography(
                        opener,
                        datafile_key,
                        topo_meta,
                        ignore_filters=ignore_filters,
                    )
                ]

                # Enumerate the measurement for ingestion. The member name is
                # the measurement name from the metadata, falling back to the
                # name of the data file inside the archive.
                member_name = topo_meta.get("name")
                if not member_name:
                    member_name = os.path.basename(datafiles[datafile_key])
                try:
                    member_metadata = TopographyMeta.model_validate(topo_meta)
                except ValidationError:
                    # Legacy metadata may not fit the schema; the member is
                    # still enumerated, just without validated metadata.
                    member_metadata = None
                members += [
                    ContainerMember(
                        name=member_name,
                        open=opener,
                        visible=True,
                        metadata=member_metadata,
                    )
                ]

            self._containers += [LazySurfaceContainer(readers, info=surf_meta)]
            self._members += [members]

    @property
    def nb_containers(self):
        """Number of surfaces stored in this container file"""
        return len(self._containers)

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

    def members(self, index=0):
        """
        Enumerate the raw data files (measurements) stored in this container.
        See :meth:`ContainerReaderBase.members`.
        """
        return self._members[index]


def write_containers(containers, fn):
    """
    Write multiple surface containers into a ZIP file.

    The container metadata is written twice, with identical content: as the
    canonical `index.json` (validated against
    :class:`SurfaceTopography.Container.IO.Schema.ContainerMeta`) and as the
    legacy `meta.yml`, which older readers understand.

    Parameters
    ----------
    containers : list of :obj:`SurfaceContainer`s
        Containers to be written to the ZIP file.
    fn : str or stream
        File or stream to write the ZIP-container to.
    """

    # This is adapted from TopoBank

    surface_metas = []
    counter = 0

    with ZipFile(fn, mode="w") as zf:
        #
        # Add meta data and topography files for all given surfaces
        #
        for surface_index, surface_container in enumerate(containers):
            topography_metas = []

            for topography in surface_container:
                # Create unique file names for the data files by simply appending a counter
                topofile_name = f"topography{counter}.nc"
                counter += 1

                # Most of the metadata (heights, sizes, units, filters) lives
                # in the NetCDF file itself (serialized as JSON); the schema
                # entry carries what is needed to locate and identify it.
                topo_meta = TopographyMeta(
                    name=str(topography.info.get("name", topofile_name)),
                    datafile=DatafileMeta(squeezed_netcdf=topofile_name),
                    size=[float(s) for s in topography.physical_sizes],
                    unit=str(topography.unit) if topography.unit is not None else None,
                    is_periodic=bool(topography.is_periodic),
                )

                # Add topography file as NetCDF to the ZIP archive
                with tempfile.TemporaryFile() as f:
                    topography.to_netcdf(f)
                    f.seek(0)
                    zf.writestr(topofile_name, f.read())

                topography_metas.append(topo_meta)

            surface_info = getattr(surface_container, "info", None) or {}
            surface_metas.append(
                SurfaceMeta(
                    name=str(surface_info.get("name", f"surface{surface_index}")),
                    topographies=topography_metas,
                )
            )

        #
        # Add metadata files. `index.json` is the schema-validated source of
        # truth; `meta.yml` carries the same content for older readers.
        #
        metadata = ContainerMeta(
            versions={"SurfaceTopography": __version__},
            surfaces=surface_metas,
            created_at=str(datetime.now()),
        )

        zf.writestr(
            CONTAINER_METADATA_FILENAME,
            metadata.model_dump_json(by_alias=True, exclude_none=True, indent=2),
        )
        zf.writestr(
            LEGACY_METADATA_FILENAME,
            yaml.dump(metadata.model_dump(by_alias=True, exclude_none=True, mode="json")),
        )

        #
        # Add a Readme file
        #
        readme_txt = textwrap.dedent(
            f"""
        Contents of this ZIP archive
        ============================
        This archive contains {len(containers)} surface(s). Each surface is a
        collection of individual topography measurements.
        In total {sum(len(x) for x in containers)} topography measurements are included.

        The meta data for the surfaces and the individual topographies
        can be found in the auxiliary files 'index.json' and 'meta.yml'
        and within the NetCDF topographies themselves. The NetCDF
        topographies contain metadata as [JSON](https://www.json.org/).

        Version information
        ===================
        SurfaceTopography: {__version__}
        """
        )
        zf.writestr("README.txt", textwrap.dedent(readme_txt))


SurfaceContainer.register_function(
    "to_zip", lambda container, fn: write_containers([container], fn)
)
