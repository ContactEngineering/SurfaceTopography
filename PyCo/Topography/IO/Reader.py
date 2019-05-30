#
# Copyright 2019 Lars Pastewka
#           2019 Kai Haase
#           2019 Antoine Sanner
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

    def __init__(self, nb_grid_pts=None, size=None, info={}):
        self._nb_grid_pts = nb_grid_pts
        self._size = size
        self._info = info
        self._default_channel=0

    @property
    def channels(self):
        """
        List of dictionnaries describing the available channels

        The dictionary at least contains the fields
        ["name", "height_scale_factor", "unit"]

        Returns
        -------
        list of dicts

        """
        channelinfo={"name": "NoName",
                     "nb_grid_pts":self._nb_grid_pts,
                     "height_scale_factor": 1.,
                     "unit": "",
                     "physical_sizes": None}

        channelinfo.update(self._info)
        return [channelinfo]

    @property
    def default_channel(self):
        """
        Index of the default_channel
        Returns
        -------

        """
        return self._default_channel

    @property
    def nb_grid_pts(self):
        return self._nb_grid_pts

    @property
    def physical_sizes(self):
        return self._size

    @property
    def info(self):
        return self._info

    def _process_size(self, size):
        if self.physical_sizes is None:
            if size is None:
                raise ValueError("physical_sizes could not be extracted from file, you should provide it")
        else:
            if size is None:
                size = self.physical_sizes
            elif [s == ss for s, ss in zip(size, self.physical_sizes)] != [True, True]: # both sizes are defined
                warnings.warn("a physical_sizes different from the specified value"
                              "was present in the file ({})."
                              "we will use the specified value".format(self.physical_sizes, size))
        return size

    def _process_info(self, info):
        newinfo = self.info.copy()
        newinfo.update(info)
        return newinfo

    @abc.abstractmethod
    def topography(self, size=None, channel=None):
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
