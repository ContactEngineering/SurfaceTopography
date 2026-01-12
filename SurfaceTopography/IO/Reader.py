#
# Copyright 2019-2023 Lars Pastewka
#           2019-2021 Michael Röttger
#           2019-2021 Antoine Sanner
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

import abc
import os
from enum import Enum

import numpy as np

from ..Exceptions import MetadataAlreadyFixedByFile
from ..Metadata import InfoModel
from ..UniformLineScanAndTopography import Topography
from .binary import AttrDict, LayoutWithNameBase
from .common import OpenFromAny


class MagicMatch(Enum):
    """Result of magic-based file format check."""

    YES = "yes"  # Magic matches, this reader should work
    NO = "no"  # Magic does NOT match, skip this reader
    MAYBE = "maybe"  # Cannot determine from magic alone


class DataKind(Enum):
    """Kind of data stored in a channel."""

    HEIGHT = "height"          # Topography height data
    VOLTAGE = "voltage"        # Piezo voltage, bias voltage, etc.
    CURRENT = "current"        # Tunneling current (STM), etc.
    PHASE = "phase"            # AFM phase data
    AMPLITUDE = "amplitude"    # AFM amplitude data
    ERROR = "error"            # Error signal
    DEFLECTION = "deflection"  # Cantilever deflection
    FRICTION = "friction"      # Friction/lateral force
    OTHER = "other"            # Generic non-height data


class ChannelInfo:
    """
    Information on topography channels contained within a file.
    """

    def __init__(
        self,
        reader,
        index,
        name=None,
        dim=None,
        nb_grid_pts=None,
        physical_sizes=None,
        height_scale_factor=None,
        periodic=None,
        uniform=None,
        undefined_data=None,
        unit=None,
        data_kind=None,
        data_unit=None,
        info={},
        tags={},
    ):
        """
        Initialize the channel. Use as much information from the file as
        possible by passing it in the keyword arguments. Keyword arguments
        can be None if the information cannot be determined. (This is the
        default for all keyword arguments.)

        Arguments
        ---------
        reader : :obj:`ReaderBase`
            Reader instance this channel is coming from.
        index : int
            Index of channel in the file, where zero is the first channel.
        name : str, optional
            Name of the channel. If no name is given, "channel <index>" will
            be used, where "<index>" is replaced with the index.
        dim : int, optional
            Number of dimensions.
        nb_grid_pts : tuple of ints, optional
            Number grid points in each dimension.
        physical_sizes : tuple of floats, optional
            Physical dimensions.
        height_scale_factor: float, optional
            Number by which all heights have been multiplied.
        periodic : bool, optional
            Whether the SurfaceTopography should be interpreted as one period of
            a periodic surface. This will affect the PSD and autocorrelation
            calculations (windowing).
        uniform : bool, optional
            Data is uniform.
        has_undefined_data : bool, optional
            Underlying data has missing/undefined points.
        unit : str, optional
            Length unit of measurement (lateral dimensions).
        data_kind : DataKind, optional
            Kind of data stored in this channel (HEIGHT, VOLTAGE, etc.).
            Defaults to HEIGHT if not specified.
        data_unit : str, optional
            Unit for data values (z-axis). For height channels, this is
            typically the same as unit. For non-height channels, this could
            be 'V', 'A', 'deg', etc.
        info : dict, optional
            Meta data found in the file. (Default: {})
        tags : dict, optional
            Additional meta data required internally by the reader
        """
        self._reader = reader
        self._index = int(index)
        self._name = "channel {}".format(self._index) if name is None else str(name)
        self._dim = dim
        self._nb_grid_pts = (
            None if nb_grid_pts is None else tuple(np.ravel(nb_grid_pts))
        )
        self._physical_sizes = (
            None if physical_sizes is None else tuple(np.ravel(physical_sizes))
        )
        self._height_scale_factor = height_scale_factor
        self._periodic = periodic
        self._uniform = uniform
        self._undefined_data = undefined_data
        self._unit = unit
        self._data_kind = data_kind if data_kind is not None else DataKind.HEIGHT
        self._data_unit = data_unit
        # These will be set by reader._finalize_channels()
        self._channel_id = None
        self._height_index = None
        if info is None:
            self._info = InfoModel()
        else:
            self._info = InfoModel(**info)
        if tags is None:
            self._tags = {}
        else:
            self._tags = tags.copy()

    def __eq__(self, other):
        # We do not compare tags, channel_id, or height_index
        return (
            self.index == other.index
            and self.name == other.name
            and self.dim == other.dim
            and self.nb_grid_pts == other.nb_grid_pts
            and self.physical_sizes == other.physical_sizes
            and self.height_scale_factor == other.height_scale_factor
            and self.is_periodic == other.is_periodic
            and self.is_uniform == other.is_uniform
            and self.has_undefined_data == other.has_undefined_data
            and self.unit == other.unit
            and self.data_kind == other.data_kind
            and self.data_unit == other.data_unit
            and self.info == other.info
        )

    def topography(
        self,
        physical_sizes=None,
        height_scale_factor=None,
        unit=None,
        info={},
        periodic=False,
        subdomain_locations=None,
        nb_subdomain_grid_pts=None,
    ):
        """
        Returns an instance of a subclass of :obj:`HeightContainer` that
        contains the topography data. The method allows to override data
        found in the data file.

        Arguments
        ---------
        physical_sizes : tuple of floats, optional
            Physical size of the topography. It is necessary to specify this
            if no physical size is found in the data file. If there is a
            physical size, then this parameter will override the physical
            size found in the data file.
        height_scale_factor : float, optional
            Factor by which the heights should be multiplied.
            This parameter is only available for file formats that do not provide
            metadata information about units and associated length scales.
        unit : str, optional
            Length unit of measurement.
        info : dict, optional
            This dictionary will be appended to the info dictionary returned
            by the reader.
        periodic : bool, optional
            Whether the SurfaceTopography should be interpreted as one period of
            a periodic surface. This will affect the PSD and autocorrelation
            calculations (windowing)
        subdomain_locations : tuple of ints, optional
            Origin (location) of the subdomain handled by the present MPI
            process.
        nb_subdomain_grid_pts : tuple of ints, optional
            Number of grid points within the subdomain handled by the present
            MPI process.

        Returns
        -------
        topography : subclass of :obj:`HeightContainer`
            The object containing the actual topography data.
        """
        return self._reader.topography(
            channel_index=self._index,
            physical_sizes=physical_sizes,
            height_scale_factor=height_scale_factor,
            unit=unit,
            info=info,
            periodic=periodic,
            subdomain_locations=subdomain_locations,
            nb_subdomain_grid_pts=nb_subdomain_grid_pts,
        )

    @property
    def index(self):
        """Unique integer channel index."""
        return self._index

    @property
    def name(self):
        """
        Name of the channel.
        Can be used in a UI for identifying a channel.
        """
        return self._name

    @property
    def dim(self):
        """
        1 for line scans and 2 for topography maps.
        """
        return self._dim

    @property
    def nb_grid_pts(self):
        """
        Number of grid points in each direction, either 1 or 2 elements
        depending on the dimension of the topography.
        """
        return self._nb_grid_pts

    @property
    def physical_sizes(self):
        """
        If the physical size can be determined from the file, a tuple is
        returned. The tuple has 1 or 2 elements (size_x, size_y), depending
        on the dimension of the topography.

        If the physical size can not be determined from the file, then
        None is returned.

        Note that the physical sizes obtained from the file can be
        overwritten by passing a `physical_sizes` argument to the
        `topography` method that returns the topography object.
        """
        return self._physical_sizes

    @physical_sizes.setter
    def physical_sizes(self, value):
        self._physical_sizes = value

    @property
    def height_scale_factor(self):
        """
        If a height scale factor can be determined from the file, a float is
        returned.

        If no height scale factor can be determined from the file, then
        None is returned.

        Note that the height scale factor obtained from the file can be
        overwritten by passing a `height_scale_factor` argument to the
        `topography` method that returns the topography object.
        """
        return self._height_scale_factor

    @height_scale_factor.setter
    def height_scale_factor(self, value):
        self._height_scale_factor = value

    @property
    def is_periodic(self):
        """
        Return whether the topography is periodically repeated at the
        boundaries.
        """
        return self._periodic

    @property
    def is_uniform(self):
        """
        Return whether the topography is uniform.
        """
        return self._uniform

    @property
    def has_undefined_data(self):
        """
        Return whether the topography has undefined data.
        """
        return self._undefined_data

    @property
    def pixel_size(self):
        """
        The pixel size is returned as the phyiscal size divided by the number
        of grid points. If the physical size can be determined from the file,
        a tuple is returned. The tuple has 1 or 2 elements (size_x, size_y),
        depending on the dimension of the topography.

        If the physical size can not be determined from the file, then
        None is returned.

        Note that the physical sizes obtained from the file can be overwritten
        by passing a `physical_sizes` argument to the `topography` method that
        returns the topography object. Overriding the physical size will also
        affect the pixel size.
        """
        if self._physical_sizes is None or self._nb_grid_pts is None:
            return None
        else:
            return tuple(np.array(self._physical_sizes) / np.array(self._nb_grid_pts))

    @property
    def area_per_pt(self):
        """
        The area per point is returned as the product over the pixel size
        tuple. If `pixel_size` returns None, than also `area_per_pt` returns
        None.
        """
        if self.pixel_size is None:
            return None
        else:
            return np.prod(self.pixel_size)

    @property
    def unit(self):
        """
        Length unit for lateral dimensions (x, y).

        This is the unit for physical_sizes. For height channels, this is
        typically also the unit of the height data. For non-height channels,
        see data_unit for the unit of the data values.
        """
        return self._unit

    @property
    def lateral_unit(self):
        """
        Length unit for lateral dimensions (x, y).

        Alias for `unit` property.
        """
        return self._unit

    @property
    def data_kind(self):
        """
        Kind of data stored in this channel.

        Returns
        -------
        DataKind
            The kind of data (HEIGHT, VOLTAGE, CURRENT, PHASE, etc.)
        """
        return self._data_kind

    @property
    def data_unit(self):
        """
        Unit for data values (z-axis).

        For height channels, this returns the lateral unit if no specific
        data unit was provided (since height is typically in the same unit
        as lateral dimensions). For non-height channels, this could be
        'V', 'A', 'deg', etc.

        Returns
        -------
        str or None
            Unit string for the data values.
        """
        if self._data_unit is None and self._data_kind == DataKind.HEIGHT:
            return self._unit
        return self._data_unit

    @property
    def is_height_channel(self):
        """
        Return whether this channel contains height data.

        Returns
        -------
        bool
            True if this is a height channel, False otherwise.
        """
        return self._data_kind == DataKind.HEIGHT

    @property
    def channel_id(self):
        """
        Stable unique string identifier for this channel.

        This ID is stable across different reads of the same file and does
        not depend on channel ordering. Format is '{name}' if the name is
        unique, or '{name}#{n}' if there are multiple channels with the
        same name.

        Returns
        -------
        str
            Stable channel identifier.
        """
        if self._channel_id is None:
            # Compute channel_id lazily by looking at all channels
            self._compute_id_and_height_index()
        return self._channel_id

    def _compute_id_and_height_index(self):
        """Compute channel_id and height_index by looking at reader's channels."""
        channels = self._reader.channels
        # Count channels with the same name before this one
        name_counts = {}
        height_idx = 0
        for ch in channels:
            ch_name = ch.name
            if ch_name not in name_counts:
                name_counts[ch_name] = 0
            if ch is self:
                # Found ourselves
                count = name_counts[ch_name]
                if count == 0:
                    # Check if there are more channels with this name after us
                    has_duplicate = any(
                        c.name == ch_name for c in channels if c is not self
                    )
                    if has_duplicate:
                        self._channel_id = f"{ch_name}#1"
                    else:
                        self._channel_id = ch_name
                else:
                    self._channel_id = f"{ch_name}#{count + 1}"
                if self.is_height_channel:
                    self._height_index = height_idx
                break
            name_counts[ch_name] += 1
            if ch.is_height_channel:
                height_idx += 1

    @property
    def height_index(self):
        """
        Index among height channels only.

        This provides backwards compatibility for code that expects channels
        to be indexed only among height data. Non-height channels return None.

        Returns
        -------
        int or None
            Index among height channels, or None for non-height channels.
        """
        if self._height_index is None and self.is_height_channel:
            # Compute lazily if not set
            self._compute_id_and_height_index()
        return self._height_index

    @property
    def info(self) -> dict:
        """
        A dictionary containing additional information (metadata) not used by
        SurfaceTopography itself, but required by third-party application.
        """
        return self._info.model_dump(exclude_none=True)

    @property
    def tags(self):
        """
        A dictionary containing additional information (metadata) used
        internally by the reader.
        """
        return self._tags

    @property
    def reader(self):
        return self._reader

    @reader.setter
    def reader(self, value):
        self._reader = value


class ReaderBase(metaclass=abc.ABCMeta):
    """
    Base class for topography readers. These are object that allow to open a
    file (from filename or stream object), inspect its metadata and then
    request to load a topography from it. Metadata is typically extracted
    without reading the full file.

    Readers should adhere to the following design rules:
    1. Opening a file should be fast and therefore not read the whole data.
       The data is read only when requesting it via the `topography` method.
    2. Data corruption must be detected when opening the file. The
       `topography` method must not fail because of file corruption issues.
    These rules are important to allow smooth operation of the readers in
    the web application `TopoBank`.
    """

    _format = None  # Short internal format string, e.g. 'di', 'sur', etc.
    _mime_types = None  # MIME type, see Gwyddion's mime specification
    _file_extensions = None  # List of common file extensions, without the '.'

    _name = None
    _description = None

    _default_channel_index = 0

    @classmethod
    def format(cls):
        """
        Short string identifier for this file format. Identifier must be
        unique and is typically equal to the file extension of this format.
        """
        if cls._format is None:
            raise RuntimeError("Reader does not provide a format string")
        return cls._format

    @classmethod
    def mime_types(cls):
        """
        MIME types supported by this reader.
        """
        if cls._mime_types is None:
            raise RuntimeError("Reader does not provide MIME types")
        return cls._mime_types

    @classmethod
    def file_extensions(cls):
        """
        A list of typical file extensions for this reader. Can be None if
        there are no typical file extensions.
        """
        if cls._file_extensions is None:
            raise RuntimeError("Reader does not provide file extensions")
        return cls._file_extensions

    @classmethod
    def name(cls):
        """
        Short name of this file format.
        """
        if cls._name is None:
            raise RuntimeError("Reader does not provide a name")
        return cls._name

    @classmethod
    def description(cls):
        """
        Long description of this file format. Should be formatted as markdown.
        """
        if cls._description is None:
            raise RuntimeError("Reader does not provide a description string")
        return cls._description

    @classmethod
    def can_read(cls, buffer: bytes) -> MagicMatch:
        """
        Check if this reader can handle a file based on magic bytes.

        This method performs a fast check using the first N bytes of a file
        to determine if this reader can handle the format. It is used to
        quickly reject incompatible formats during auto-detection.

        Parameters
        ----------
        buffer : bytes
            First N bytes of the file (typically 512 bytes).

        Returns
        -------
        MagicMatch
            YES if magic matches and this reader should work.
            NO if magic does NOT match and this reader should be skipped.
            MAYBE if cannot determine from magic alone (default).
        """
        return MagicMatch.MAYBE

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        pass

    def _get_channel_by_id(self, channel_id):
        """
        Find a channel by its stable channel ID.

        Parameters
        ----------
        channel_id : str
            The stable channel identifier.

        Returns
        -------
        ChannelInfo
            The channel with the given ID.

        Raises
        ------
        ValueError
            If no channel with the given ID is found.
        """
        for ch in self.channels:
            if ch.channel_id == channel_id:
                return ch
        raise ValueError(f"No channel with id '{channel_id}' found.")

    def _get_channel_by_height_index(self, height_index):
        """
        Find a channel by its height index (index among height channels only).

        Parameters
        ----------
        height_index : int
            Index among height channels.

        Returns
        -------
        ChannelInfo
            The height channel at the given index.

        Raises
        ------
        ValueError
            If no height channel with the given index is found.
        """
        for ch in self.channels:
            if ch.height_index == height_index:
                return ch
        raise ValueError(f"No height channel with height_index {height_index} found.")

    def height_index_to_channel_id(self, height_index):
        """
        Convert a height channel index to a stable channel ID.

        This utility function is useful for migrating databases that stored
        channel references using the old height-only index system to the new
        stable channel_id system.

        Parameters
        ----------
        height_index : int
            Index among height channels (0-based).

        Returns
        -------
        str
            The stable channel_id for the height channel at that index.

        Raises
        ------
        ValueError
            If no height channel exists at the given index.

        Examples
        --------
        >>> reader = open_topography("scan.ibw")
        >>> # Migrate old database entry
        >>> old_index = 0  # stored in database
        >>> new_id = reader.height_index_to_channel_id(old_index)
        >>> # Store new_id in database instead
        """
        channel = self._get_channel_by_height_index(height_index)
        return channel.channel_id

    def channel_id_to_height_index(self, channel_id):
        """
        Convert a stable channel ID to a height channel index.

        This utility function can be used to convert channel IDs back to
        height indices for backwards compatibility with systems that
        require numeric indices.

        Parameters
        ----------
        channel_id : str
            The stable channel identifier.

        Returns
        -------
        int
            The height index for the channel with that ID.

        Raises
        ------
        ValueError
            If no channel with the given ID is found, or if the channel
            is not a height channel.

        Examples
        --------
        >>> reader = open_topography("scan.ibw")
        >>> height_idx = reader.channel_id_to_height_index("Height")
        """
        channel = self._get_channel_by_id(channel_id)
        if not channel.is_height_channel:
            raise ValueError(
                f"Channel '{channel_id}' is not a height channel and has no "
                f"height_index. Its data_kind is {channel.data_kind.value}."
            )
        return channel.height_index

    def _resolve_channel(self, channel_index=None, channel_id=None,
                         height_channel_index=None):
        """
        Resolve channel from any of the three selection methods.

        This helper method validates that at most one selection method is used
        and returns the appropriate channel. If no selection is provided,
        returns the default channel.

        Parameters
        ----------
        channel_index : int, optional
            Index into the full channels list.
        channel_id : str, optional
            Stable channel identifier string.
        height_channel_index : int, optional
            Index among height channels only.

        Returns
        -------
        tuple
            (channel_info, resolved_channel_index) where channel_info is the
            ChannelInfo object and resolved_channel_index is the index into
            the full channels list.

        Raises
        ------
        ValueError
            If more than one selection method is provided.
        """
        # Count how many selection methods are provided
        n_specified = sum(x is not None for x in
                          [channel_index, channel_id, height_channel_index])
        if n_specified > 1:
            raise ValueError(
                "Only one of channel_index, channel_id, or height_channel_index "
                "should be specified, but multiple were provided."
            )

        if channel_id is not None:
            # Look up by stable ID
            channel_info = self._get_channel_by_id(channel_id)
            resolved_index = self.channels.index(channel_info)
        elif height_channel_index is not None:
            # Look up by height channel index (backwards compatible)
            channel_info = self._get_channel_by_height_index(height_channel_index)
            resolved_index = self.channels.index(channel_info)
        elif channel_index is not None:
            # Direct index into channels list
            resolved_index = channel_index
            channel_info = self.channels[resolved_index]
        else:
            # Use default channel
            resolved_index = self._default_channel_index
            channel_info = self.channels[resolved_index]

        return channel_info, resolved_index

    @property
    @abc.abstractmethod
    def channels(self):
        """
        Returns a list of :obj:`ChannelInfo`s describing the available data
        channels.
        """
        raise NotImplementedError

    @property
    def height_channels(self):
        """
        Returns a list of :obj:`ChannelInfo`s for height channels only.

        This provides backwards compatibility for code that expects only
        height data channels.
        """
        return [ch for ch in self.channels if ch.is_height_channel]

    @property
    def default_channel(self):
        """Return the default channel. This is often the first channel with
        height information."""
        return self.channels[self._default_channel_index]

    @classmethod
    def _check_physical_sizes(cls, physical_sizes_from_arg, physical_sizes=None):
        """Handle overriding of `physical_sizes` arguments and make sure,
        that return value is not `None`.

        Parameters
        ----------
        physical_sizes_from_arg: tuple or float or None
            physical sizes given as argument when instantiating a topography from a reader
        physical_sizes: tuple or float or None
            physical sizes which were present in the file
        """
        if physical_sizes is None:
            if physical_sizes_from_arg is None:
                raise ValueError(
                    "`physical_sizes` could not be extracted from file, you must "
                    "provide it."
                )
            eff_physical_sizes = physical_sizes_from_arg
        elif physical_sizes_from_arg is not None:
            # in general this should result in an exception now
            raise MetadataAlreadyFixedByFile("physical_sizes")
        else:
            eff_physical_sizes = physical_sizes

        return eff_physical_sizes

    @abc.abstractmethod
    def topography(
        self,
        channel_index=None,
        channel_id=None,
        height_channel_index=None,
        physical_sizes=None,
        height_scale_factor=None,
        unit=None,
        info={},
        periodic=False,
        subdomain_locations=None,
        nb_subdomain_grid_pts=None,
    ):
        """
        Returns an instance of a subclass of :obj:`HeightContainer` that
        contains the topography data. Which specific type of height container
        (1D, 2D, uniform, nonuniform) is returned may depend on the file
        content and is determined dynamically.

        The returned object needs to have `physical_sizes` and periodicity
        information. If this information is not provided by the data file,
        then the user must specify it when calling this method. Conversely,
        metadata that is present in the file cannot be overridden but needs to
        lead to a `MetadataAlreadyFixedByFile` exception. The user can check
        if the metadata is present by inspecting the `ChannelInfo` of the
        respective data channel.

        There are three ways to specify which channel to load:
        1. `channel_index`: Index into the full channels list (all data types)
        2. `channel_id`: Stable string identifier (e.g., "Height", "Phase#2")
        3. `height_channel_index`: Index among height channels only (backwards
           compatible with pre-existing database indices)

        Only one of these parameters should be specified. If none are given,
        the default channel is loaded.

        Arguments
        ---------
        channel_index : int
            Index of the channel to load. See also `channels` method.
            (Default: None, which loads the default channel)
        channel_id : str
            Stable channel identifier string. This is the recommended way to
            identify channels for storage in databases as it remains stable
            even if channel ordering changes.
        height_channel_index : int
            Index among height channels only. This provides backwards
            compatibility for databases that stored channel indices before
            non-height channels were supported.
        physical_sizes : tuple of floats
            Physical size of the topography. It is necessary to specify this
            if no physical size is found in the data file. If there is a
            physical size, then this parameter will override the physical
            size found in the data file.
        height_scale_factor : float
            Override height scale factor found in the data file.
        unit : str
            Length unit.
        info : dict
            This dictionary will be appended to the info dictionary returned
            by the reader.
        periodic: bool
            Whether the SurfaceTopography should be interpreted as one period of
            a periodic surface. This will affect the PSD and autocorrelation
            calculations (windowing).
        subdomain_locations : tuple of ints
            Origin (location) of the subdomain handled by the present MPI
            process.
        nb_subdomain_grid_pts : tuple of ints
            Number of grid points within the subdomain handled by the present
            MPI process.

        Returns
        -------
        topography : subclass of :obj:`HeightContainer`
            The object containing the actual topography data.

        Raises
        ------
        MetadataAlreadyFixedByFile
            Raised if `physical_sizes`, `unit or `height_scale_factor` have
            already been defined in the file, because they should not be
            overridden by the user.
        ValueError
            Raised if more than one channel selection parameter is provided,
            or if the specified channel cannot be found.
        """
        raise NotImplementedError


class CompoundLayout(LayoutWithNameBase):
    """Declare a file layout"""

    def __init__(self, structures, name=None, context_mapper=lambda x: x):
        self._structures = structures
        self._name = name
        self._context_mapper = context_mapper

    def from_stream(self, stream_obj, context):
        local_context = AttrDict()
        for structure in self._structures:
            tmp_context = AttrDict({**local_context, "__parent__": context})
            if callable(structure):
                # This is a function that returns the respective structure
                structure = structure(tmp_context)
            local_context.update(structure.from_stream(stream_obj, tmp_context))
        name = self.name(context)
        if name is None:
            return self._context_mapper(local_context)
        else:
            return self._context_mapper({name: local_context})


class If:
    """Switch structure type dependent on data"""

    def __init__(self, *args, context_mapper=None):
        self._args = args
        self._context_mapper = context_mapper

    def name(self, context):
        nb_conditions = len(self._args) // 2
        for i in range(nb_conditions):
            if self._args[2 * i](context):
                return self._args[2 * i + 1].name(context)
        if 2 * nb_conditions == len(self._args):
            return None
        else:
            return self._args[2 * nb_conditions].name(context)

    def from_stream(self, stream_obj, context):
        if self._context_mapper is not None:
            context = self._context_mapper(context)
        nb_conditions = len(self._args) // 2
        for i in range(nb_conditions):
            if self._args[2 * i](context):
                return self._args[2 * i + 1].from_stream(stream_obj, context)
        if 2 * nb_conditions == len(self._args):
            return {}
        else:
            return self._args[2 * nb_conditions].from_stream(stream_obj, context)


class Skip:
    """
    Skips over bytes in a binary stream without storing them.

    Use this for TLV entries or file sections that should be ignored.

    Parameters
    ----------
    size : int, callable, or None
        Size of the data block in bytes. Can be:
        - An integer for fixed size
        - A callable that takes context and returns size
        - None to read size from context['_block_size'] (for TLV parsing)
    comment : str, optional
        Description of what is being skipped (for documentation).
    """

    def __init__(self, size=None, comment=None):
        self._size = size
        self._comment = comment

    def from_stream(self, stream_obj, context):
        if self._size is None:
            size = context.get('_block_size', 0)
        elif callable(self._size):
            size = self._size(context)
        else:
            size = self._size
        stream_obj.seek(size, os.SEEK_CUR)
        return {}


class SizedChunk(LayoutWithNameBase):
    def __init__(
        self,
        size,
        structure,
        mode="read-once",
        name=None,
        context_mapper=None,
        debug=False,
    ):
        """
        Declare a file portion of a specific size. This allows bounds checking,
        i.e. raising exception if too much or too little data is read.

        Parameters
        ----------
        size : function or int
            Function that returns the size of the chunk and takes as input the
            current context.
        structure : structure definition
            Definition of the structure within this chunk.
        mode : str
            Reading mode, can be 'read-once', 'skip-missing' or 'loop'.
            'read-once' and 'skip-missing' read the containing block once,
            but 'skip-missing' does not complain if this is not the complete
            chunk. 'loop' repeats the containing block until the full chunk
            has been read. (Default: 'read-once')
        name : str
            Context name of this block. Required if mode is 'loop'.
        context_mapper : callable
            Function that takes a context and returns a new context.
        debug : Bool
            Print boundary offsets of chunk. (Default: False)
        """
        self._size = size
        self._structure = structure
        self._mode = mode
        self._name = name
        self._context_mapper = context_mapper
        self._debug = debug

    def from_stream(self, stream_obj, context):
        """
        Decode stream into dictionary.

        Parameters
        ----------
        stream_obj : stream-like object
            Binary stream to decode.
        context : dict
            Dictionary with data that has been decoded at this point.

        Returns
        -------
        decoded_data : dict
            Dictionary with decoded data entries.
        """
        size = self._size(context) if callable(self._size) else self._size

        starting_position = stream_obj.tell()
        if self._debug:
            print(
                f"Sized chunk is supposed to run from {starting_position} to {starting_position + size}."
            )

        local_context = []
        local_context += [self._structure.from_stream(stream_obj, context)]
        final_position = stream_obj.tell()
        while self._mode == "loop" and final_position - starting_position < size:
            local_context += [self._structure.from_stream(stream_obj, context)]
            final_position = stream_obj.tell()

        # Check if we processed the whole chunk
        if final_position - starting_position > size:
            raise IOError(
                f"Chunk is supposed to contain {size} bytes, but "
                f"{final_position - starting_position} bytes have been decoded. "
                f"(Chunk starts at position {starting_position} and ends at {final_position}, "
                f"but is supposed to end at {starting_position + size}.)"
            )
        elif final_position - starting_position < size:
            if self._mode == "skip-missing":
                stream_obj.seek(
                    size - (final_position - starting_position), os.SEEK_CUR
                )
            else:
                raise IOError(
                    f"Chunk is supposed to contain {size} bytes, but only "
                    f"{final_position - starting_position} bytes have been decoded. "
                    f"(Chunk starts at position {starting_position} and ends at {final_position}, "
                    f"but is supposed to end at {starting_position + size}.)"
                )

        name = self.name(context)
        if self._mode != "loop":
            (local_context,) = local_context
        if self._context_mapper is None:

            def context_mapper(x):
                return x

        else:
            context_mapper = self._context_mapper
        if name is None:
            if self._mode == "loop" and self._context_mapper is None:
                raise IOError(
                    "`name` or `context_mapper` must be specified for mode 'loop'."
                )
            return context_mapper(local_context)
        else:
            return context_mapper({name: local_context})


class For(LayoutWithNameBase):
    """Repeat structure"""

    def __init__(self, range, structure, name=None):
        self._range = range
        self._structure = structure
        self._name = name

        if self._name is None:
            raise ValueError("`For` statement must have a name.")

    def from_stream(self, stream_obj, context):
        local_context = []
        if callable(self._range):
            nb_elements = self._range(context)
        else:
            nb_elements = self._range
        for i in range(nb_elements):
            local_context += [self._structure.from_stream(stream_obj, context)]
        return {self.name(context): local_context}


class While(LayoutWithNameBase):
    """Repeat as long as condition is met"""

    def __init__(self, *args, name=None):
        self._args = args
        self._name = name

        if self._name is None:
            raise ValueError("`While` statement must have a name.")

    def from_stream(self, stream_obj, context):
        while_context = []
        still_looping = True
        while still_looping:
            local_context = AttrDict()
            for arg in self._args:
                if still_looping:
                    if callable(arg):
                        still_looping = arg(
                            AttrDict({**local_context, "__parent__": context})
                        )
                    else:
                        x = arg.from_stream(
                            stream_obj,
                            AttrDict({**local_context, "__parent__": context}),
                        )
                        local_context.update(x)
            while_context += [local_context]
        return {self.name(context): while_context}


class DeclarativeReaderBase(ReaderBase):
    """
    Base class for automatic readers.
    """

    _file_layout = None

    def __init__(self, file_path):
        if self._file_layout is None:
            raise RuntimeError(
                "Please defined the file structure via the `_file_layout` class member."
            )

        self.file_path = file_path
        with OpenFromAny(self.file_path, "rb") as f:
            self._metadata = self._file_layout.from_stream(f, {})

        self._validate_metadata()

    def _validate_metadata(self):
        pass

    @property
    def metadata(self):
        return self._metadata

    def topography(
        self,
        channel_index=None,
        physical_sizes=None,
        height_scale_factor=None,
        unit=None,
        info={},
        periodic=None,
        subdomain_locations=None,
        nb_subdomain_grid_pts=None,
    ):
        if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
            raise RuntimeError("This reader does not support MPI parallelization.")

        if channel_index is None:
            channel_index = self._default_channel_index

        channels = self.channels
        if channel_index < 0 or channel_index >= len(channels):
            raise RuntimeError(
                f"Channel index is {channel_index} but must be between 0 and {len(channels) - 1}."
            )

        # Get channel information
        channel = channels[channel_index]

        if physical_sizes is None:
            physical_sizes = channel.physical_sizes
        elif channel.physical_sizes is not None:
            raise MetadataAlreadyFixedByFile("physical_sizes")

        if height_scale_factor is None:
            height_scale_factor = channel.height_scale_factor
        elif channel.height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile("height_scale_factor")

        if unit is None:
            unit = channel.unit
        elif channel.unit is not None:
            raise MetadataAlreadyFixedByFile("unit")

        with OpenFromAny(self.file_path, "rb") as f:
            height_data = channel.tags["reader"](f)

        _info = channel.info.copy()
        _info.update(info)

        topo = Topography(
            height_data,
            physical_sizes,
            unit=unit,
            periodic=False if periodic is None else periodic,
            info=_info,
        )
        return topo.scale(height_scale_factor)
