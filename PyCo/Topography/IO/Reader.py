#
# Copyright 2019 Lars Pastewka
#           2019 Antoine Sanner
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import warnings
import abc


class ReaderBase(metaclass=abc.ABCMeta):
    def __init__(self, nb_grid_pts=None, physical_sizes=None, periodic=False, info={}):
        self._nb_grid_pts = nb_grid_pts
        self._size = physical_sizes
        self._periodic = periodic
        self._info = info
        self._default_channel = 0

    @property
    def channels(self):
        """
        Returns a list of dictionaries describing the available data channels.

        Each dict has as least the following elements:

        name : str
            Name of the channel. If no name is found in the file,
            "Default" is used. Can be used in a UI for identifying a channel.

        nb_grid_pts : tuple of ints
            Number of grid points in each direction, either 1 or 2 elements
            depending on the dimension of the topography.

        physical_sizes : tuple of floats or None
            If the physical size can be determined from the file, a tuple is
            returned. The tuple has 1 or 2 elements (size_x, size_y), depending
            on the dimension of the topography.

            If the physical size can not be determined from the file, then
            None is returned.

            Note that the physical sizes obtained from the file can be
            overwritten by passing a `physical_sizes` argument to the
            `topography` method that returns the topography object.

        unit : str or tuple (str,??)
            The unit is either a string or tuple (str, ??).

            [I don't know what the tuple means and what the
            type of the second argument can be. I only know I shouldn't
            take those topographies for topobank.]

            This is the unit of the physical size (if given in the file)
            and the units of the heights. Please also see "height_scale_factor"
            below.

         height_scale_factor : float
             Factor which was used to scale the raw numbers from file (which
             can be voltages or some other quantity) to heights with the given
             'unit'. Use `topography.heights()` in order to get the heights.
        """
        channelinfo = {"name": "Default",
                       "nb_grid_pts": self._nb_grid_pts,
                       "height_scale_factor": 1.,
                       "unit": "",
                       "physical_sizes": None}

        channelinfo.update(self._info)
        return [channelinfo]

    @property
    def default_channel(self):
        """Return the index of the default channel."""
        return self._default_channel

    @property
    def nb_grid_pts(self):
        """Return the number of grid points of the topography."""
        return self._nb_grid_pts

    @property
    def physical_sizes(self):
        """Return the physical sizes of the topography."""
        return self._size

    @property
    def dim(self):
        """Returns 1 for line scans and 2 for topography maps."""
        raise len(self._nb_grid_pts)

    @property
    def is_periodic(self):
        """Return whether the topography is periodically repeated at the boundaries."""
        return self._periodic

    @property
    def info(self):
        """
        Return the info dictionary. The info dictionary contains auxiliary data
        found in the topography data file but not directly used by PyCo.

        The dictionary can contain any type of information. There are a few
        standardized keys, listed in the following.

        Standardized keys:
        unit : str
            Unit of the topography. The unit information applies to the lateral
            units (the physical size) as well as to heights units. Examples:
            'µm', 'nm'.
        """
        return self._info

    def _process_size(self, size):
        if self.physical_sizes is None:
            if size is None:
                raise ValueError("physical_sizes could not be extracted from file, you should provide it")
        else:
            if size is None:
                size = self.physical_sizes
            elif tuple(size) != tuple(self.physical_sizes):
                warnings.warn("A physical size different from the value specified when calling the reader "
                              "was present in the file. We will ignore the value given in the file."
                              "Specified values: {}; Values from file: {}".format(self.physical_sizes, size))
        return size

    def _process_info(self, info):
        newinfo = self.info.copy()
        newinfo.update(info)
        return newinfo

    @abc.abstractmethod
    def topography(self, physical_sizes=None, channel=None):
        """
        returns a `Topography` instance containing the data

        Returns
        -------
        """
        raise NotImplementedError


class ReadFileError(Exception):
    pass


class UnknownFileFormatGiven(ReadFileError):
    pass


class CannotDetectFileFormat(ReadFileError):
    """
    Raised when no reader is able to open_topography the file
    """


class FileFormatMismatch(ReadFileError):
    """
    Raised when the reader cannot interpret the file at all
    (obvious for txt vs binary, but holds also for a header)
    """
    pass


class CorruptFile(ReadFileError):
    """
    Raised when the reader identifies the file format as matching,
    but there is a mistake, for example the number of points doesn't match
    """
    pass
