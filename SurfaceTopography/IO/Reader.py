#
# Copyright 2019-2021 Lars Pastewka
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

import numpy as np

from ..Exceptions import MetadataAlreadyFixedByFile


class ChannelInfo:
    """
    Information on topography channels contained within a file.
    """

    def __init__(self, reader, index, name=None, dim=None, nb_grid_pts=None, physical_sizes=None,
                 height_scale_factor=None, periodic=None, uniform=None, undefined_data=None, unit=None, info={}):
        """
        Initialize the channel. Use as many information from the file as
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
            Length unit of measurement.
        info : dict, optional
            Meta data found in the file. (Default: {})
        """
        self._reader = reader
        self._index = int(index)
        self._name = "channel {}".format(self._index) \
            if name is None else str(name)
        self._dim = dim
        self._nb_grid_pts = None \
            if nb_grid_pts is None else tuple(np.ravel(nb_grid_pts))
        self._physical_sizes = None \
            if physical_sizes is None else tuple(np.ravel(physical_sizes))
        self._height_scale_factor = height_scale_factor
        self._periodic = periodic
        self._uniform = uniform
        self._undefined_data = undefined_data
        self._unit = unit
        if info is None:
            self._info = None
        else:
            self._info = info.copy()

    def topography(self, physical_sizes=None, height_scale_factor=None, unit=None, info={}, periodic=False,
                   subdomain_locations=None, nb_subdomain_grid_pts=None):
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
            nb_subdomain_grid_pts=nb_subdomain_grid_pts)

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
            return tuple(np.array(self._physical_sizes) /
                         np.array(self._nb_grid_pts))

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
        Length unit.
        """
        return self._unit

    @property
    def info(self):
        """
        A dictionary containing additional information (metadata) not used by
        SurfaceTopography itself, but required by third-party application.

        Presently, the following entries have been standardized:
        'height_scale_factor':
            Factor which was used to scale the raw numbers from file (which
            can be voltages or some other quantity actually acquired in the
            measurement technique) to heights with the given 'unit'.
        """
        info = self._info.copy()
        if self.unit is not None:
            info.update(dict(unit=self.unit))
        return info


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

    _format = None
    _name = None
    _description = None

    _default_channel_index = 0

    @classmethod
    def format(cls):
        """
        String identifier for this file format. Identifier must be unique and
        is typically equal to the file extension of this format.
        """
        if cls._format is None:
            raise RuntimeError('Reader does not provide a format string')
        return cls._format

    @classmethod
    def name(cls):
        """
        Short name of this file format.
        """
        if cls._name is None:
            raise RuntimeError('Reader does not provide a name')
        return cls._name

    @classmethod
    def description(cls):
        """
        Long description of this file format. Should be formatted as markdown.
        """
        if cls._description is None:
            raise RuntimeError('Reader does not provide a description string')
        return cls._description

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        pass

    @property
    @abc.abstractmethod
    def channels(self):
        """
        Returns a list of :obj:`ChannelInfo`s describing the available data
        channels.
        """
        raise NotImplementedError

    @property
    def default_channel(self):
        """Return the default channel. This is often the first channel with
        height information."""
        return self.channels[self._default_channel_index]

    @classmethod
    def _check_physical_sizes(cls, physical_sizes_from_arg,
                              physical_sizes=None):
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
                raise ValueError("`physical_sizes` could not be extracted from file, you must provide it.")
            eff_physical_sizes = physical_sizes_from_arg
        elif physical_sizes_from_arg is not None:
            # in general this should result in an exception now
            raise MetadataAlreadyFixedByFile('physical_sizes')
        else:
            eff_physical_sizes = physical_sizes

        return eff_physical_sizes

    @abc.abstractmethod
    def topography(self, channel_index=None, physical_sizes=None, height_scale_factor=None, unit=None, info={},
                   periodic=False, subdomain_locations=None, nb_subdomain_grid_pts=None):
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

        Arguments
        ---------
        channel_index : int
            Index of the channel to load. See also `channels` method.
            (Default: None, which load the default channel)
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
        """
        raise NotImplementedError
