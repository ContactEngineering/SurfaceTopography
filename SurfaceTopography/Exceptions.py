#
# Copyright 2021, 2023 Lars Pastewka
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
This module contains most exceptions used in SurfaceTopography.
"""


class CannotPerformAnalysisError(Exception):
    """
    Exception raised when an analysis cannot be performed.
    """
    pass


class NoReliableDataError(CannotPerformAnalysisError):
    """
    Exception indicates that an analysis has no reliable data and would need
    to return an empty result.
    """
    pass


class ReentrantDataError(CannotPerformAnalysisError):
    """
    Exception indicates that the underlying data is reentrant and that the
    analysis function does not work on reentrant data.
    """
    pass


class UndefinedDataError(CannotPerformAnalysisError):
    """
    Exception indicates that the underlying data has missing (undefined) data
    points and that the analysis function cannot handle missing data
    """
    pass


# I/O exceptions

class UnknownFileFormat(Exception):
    """
    Raised when an unknown file format identifier was provided to a reader
    function.
    """
    pass


class ReadFileError(Exception):
    pass


class CannotDetectFileFormat(ReadFileError):
    """
    Raised when no reader is able to open_topography the file
    """
    pass


class FileFormatMismatch(ReadFileError):
    """
    Raised when the reader cannot interpret the file at all
    (obvious for txt vs binary, but holds also for a header)
    """
    pass


class UnsupportedFormatFeature(ReadFileError):
    """
    Raised when the reader detected that this is the correct format, but a
    file feature was encountered that is not implemented
    """
    pass


class CorruptFile(ReadFileError):
    """
    Raised when the reader identifies the file format as matching,
    but there is a mistake, for example the number of points doesn't match
    """
    pass


class MetadataAlreadyFixedByFile(ReadFileError):
    """
    Raised when instantiating a topography from a reader,
    given metadata which cannot be overridden, because
    it is already fixed by the file contents.
    """

    def __init__(self, kw, alt_msg=None):
        """
        Parameters
        ----------
        kw: str
            Name of the keyword argument to .topography().
        alt_msg: str or None
            If not None, use this as error message instead of
            the default one.
        """
        self._kw = kw
        self._alt_msg = alt_msg

    def __str__(self):
        if self._alt_msg:
            return self._alt_msg
        else:
            return f"Value for keyword '{self._kw}' is already fixed by file contents and cannot be overridden"
