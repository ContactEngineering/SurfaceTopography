#
# Copyright 2026 Lars Pastewka
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

"""
Pydantic schema for the surface-container metadata file.

A surface container is a ZIP archive bundling one or more digital surface twins
(surfaces), each holding any number of topography measurements, together with
their data files and a single metadata file.

There is exactly one source of truth for that metadata: ``index.json``. This
module holds the canonical schema; TopoBank re-exports it (as
``topobank.manager.container_schema``) and uses it for both export and import,
and :class:`SurfaceTopography.Container.IO.CE.CEReader` validates ``index.json``
against it when reading a container. Archives produced by older versions of
TopoBank stored a similar structure as a YAML file named ``meta.yml``; those
are still accepted on import (see :func:`load_container_metadata`).

Serialization rule
-------------------
``index.json`` is produced with ``model_dump(by_alias=True, exclude_none=True)``:
any field whose value is ``None`` is omitted entirely. In particular the
``orcid`` of an author and the ``squeezed-netcdf`` data file are only written
when present.
"""

from datetime import date
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Canonical name of the metadata file inside a container archive.
CONTAINER_METADATA_FILENAME = "index.json"
#: Name of the legacy YAML metadata file written by older TopoBank versions.
LEGACY_METADATA_FILENAME = "meta.yml"

# Defaults applied when importing older or partial containers. These mirror the
# database defaults of the corresponding TopoBank model fields, so a container
# that omits an optional attribute imports the same way it would have been
# created.
DEFAULT_FILL_UNDEFINED_DATA_MODE = "do-not-fill"
DEFAULT_DETREND_MODE = "center"
DEFAULT_MEASUREMENT_DATE = date(1970, 1, 1)


class CreatedBy(BaseModel):
    """Author of a surface or topography."""

    name: str
    orcid: Optional[str] = None


class PropertyMeta(BaseModel):
    """A single custom key/value property attached to a surface."""

    name: str
    value: Union[str, int, float]
    unit: Optional[str] = None


class PublicationMeta(BaseModel):
    """Publication information, present only for published surfaces."""

    url: str
    license: str
    authors: str
    version: int
    date: str
    doi_url: str = ""
    doi_state: str = ""

    @field_validator("date", mode="before")
    @classmethod
    def _stringify(cls, value):
        # YAML may parse an ISO date into a ``datetime``/``date`` object.
        return value if value is None else str(value)


class InstrumentMeta(BaseModel):
    """Instrument used to acquire a measurement."""

    name: Optional[str] = None
    type: Optional[str] = None
    parameters: dict = Field(default_factory=dict)


class DatafileMeta(BaseModel):
    """References to the data files of a single measurement inside the archive."""

    model_config = ConfigDict(populate_by_name=True)

    #: Original data file as uploaded by the user. Optional because containers
    #: written by `SurfaceTopography.Container.IO.CE.write_containers` bundle
    #: only a preprocessed NetCDF; TopoBank always writes it.
    original: Optional[str] = None
    #: Optional NetCDF 3 file with preprocessed ("squeezed") data.
    squeezed_netcdf: Optional[str] = Field(default=None, alias="squeezed-netcdf")


class TopographyMeta(BaseModel):
    """Metadata of a single topography measurement."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    datafile: DatafileMeta
    size: list[float]
    # The lateral unit is normally always set, but older archives occasionally
    # omit it; keep it optional so they still import (as the legacy code did).
    unit: Optional[str] = None
    data_source: int = 0
    has_undefined_data: Optional[bool] = None
    fill_undefined_data_mode: str = DEFAULT_FILL_UNDEFINED_DATA_MODE
    detrend_mode: str = DEFAULT_DETREND_MODE
    is_periodic: bool = False
    created_by: Optional[CreatedBy] = None
    # Absent -> default; explicit null is kept (older measurements may lack a
    # date), matching the previous import behavior.
    measurement_date: Optional[date] = DEFAULT_MEASUREMENT_DATE
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    instrument: Optional[InstrumentMeta] = None
    # Only written when the height scale is editable, i.e. not already encoded
    # in the data file itself (see GH 718).
    height_scale: Optional[float] = None


class SurfaceMeta(BaseModel):
    """Metadata of a single digital surface twin and its measurements."""

    name: str
    category: Optional[str] = None
    created_by: Optional[CreatedBy] = None
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    is_published: bool = False
    publication: Optional[PublicationMeta] = None
    properties: Optional[list[PropertyMeta]] = None
    topographies: list[TopographyMeta] = Field(default_factory=list)


class ContainerMeta(BaseModel):
    """Top-level schema of a container's ``index.json`` (or legacy ``meta.yml``)."""

    versions: dict[str, str] = Field(default_factory=dict)
    surfaces: list[SurfaceMeta] = Field(default_factory=list)
    created_at: Optional[str] = None
    # Opaque, optional metadata that TopoBank carries through verbatim without
    # interpreting it. It lets plugins (e.g. the SDS API, which owns training
    # groups) attach extra container-level information; members referenced
    # therein are expected to be indices into ``surfaces``.
    extra: Optional[dict] = None

    @field_validator("created_at", mode="before")
    @classmethod
    def _stringify(cls, value):
        # YAML may parse the timestamp into a ``datetime`` object.
        return value if value is None else str(value)


def load_container_metadata(surface_zip, unsafe_yaml_loader=False) -> ContainerMeta:
    """
    Load and validate the metadata of a surface container.

    The canonical metadata file is ``index.json``. For backward compatibility
    with archives produced by older versions of TopoBank, a legacy ``meta.yml``
    file is accepted as a fallback. In both cases the metadata is validated
    against (and returned as) a :class:`ContainerMeta` instance.

    Parameters
    ----------
    surface_zip : zipfile.ZipFile
        The container archive.
    unsafe_yaml_loader : bool, optional
        If True, use an unsafe YAML loader when reading the legacy ``meta.yml``.
        This can execute malicious code contained in the YAML file and should be
        used with caution. (Default: False)

    Returns
    -------
    ContainerMeta
        The validated container metadata.
    """
    import yaml

    names = set(surface_zip.namelist())
    if CONTAINER_METADATA_FILENAME in names:
        with surface_zip.open(CONTAINER_METADATA_FILENAME, mode="r") as f:
            return ContainerMeta.model_validate_json(f.read())
    if LEGACY_METADATA_FILENAME in names:
        with surface_zip.open(LEGACY_METADATA_FILENAME, mode="r") as f:
            yaml_loader = yaml.full_load if unsafe_yaml_loader else yaml.safe_load
            return ContainerMeta.model_validate(yaml_loader(f))
    raise RuntimeError(
        f"Container archive contains neither '{CONTAINER_METADATA_FILENAME}' nor "
        f"'{LEGACY_METADATA_FILENAME}'; cannot determine its contents."
    )
