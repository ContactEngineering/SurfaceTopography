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

import abc
import warnings


class ReaderBase(metaclass=abc.ABCMeta):
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
        Returns a list of dictionaries describing the available data channels.

        Each dict has as least the following elements:

        name : str
            Name of the channel. If no name is found in the file,
            "Default" is used. Can be used in a UI for identifying a channel.

        dim : int
            1 for line scans and 2 for topography maps.

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

        unit : str
            This is the unit of the physical size (if given in the file)
            and the units of the heights. Please also see "height_scale_factor"
            below.

         height_scale_factor : float
             Factor which was used to scale the raw numbers from file (which
             can be voltages or some other quantity) to heights with the given
             'unit'. Use `topography.heights()` in order to get the heights.
        """
        raise NotImplementedError

    @property
    def default_channel(self):
        """Return the index of the default channel."""
        return 0

    @classmethod
    def _check_physical_sizes(self, physical_sizes_from_arg, physical_sizes=None):
        if physical_sizes is None:
            if physical_sizes_from_arg is None:
                raise ValueError("physical_sizes could not be extracted from file, you should provide it")
        else:
            if physical_sizes_from_arg is None:
                physical_sizes_from_arg = physical_sizes
            elif tuple(physical_sizes_from_arg) != tuple(physical_sizes):
                warnings.warn("A physical size different from the value specified when calling the reader "
                              "was present in the file. We will ignore the value given in the file. "
                              "Specified values: {}; Values from file: {}".format(physical_sizes,
                                                                                  physical_sizes_from_arg))
        return physical_sizes_from_arg

    @abc.abstractmethod
    def topography(self, channel=None, physical_sizes=None, height_scale_factor=None, info={},
                   subdomain_locations=None, nb_subdomain_grid_pts=None):
        """
        Returns an instance of a subclass of :obj:`HeightContainer` that
        contains the topography data. The method allows to override data
        found in the data file.

        Arguments
        ---------
        channel : int
            Number of the channel to load. See also `channels` method.
        physical_sizes : tuple of floats
            Physical size of the topography. It is necessary to specify this
            if no physical size is found in the data file. If there is a
            physical size, then this parameter will override the physical
            size found in the data file.
        height_scale_factor : float
            Override height scale factor found in the data file.
        info : dict
            This dictionary will be appended to the info dictionary returned
            by the reader.
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
