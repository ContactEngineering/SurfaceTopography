#
# Copyright 2023 Lars Pastewka
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


class ContainerReaderBase(metaclass=abc.ABCMeta):
    """
    Base class for container readers. These are object that allow to open a
    file (from filename or stream object), inspect its metadata and then
    request to load a surface container from it. Metadata is typically
    extracted without reading the full file.

    Readers should adhere to the following design rules:
    1. Opening a file should be fast and therefore not read the whole data.
       The data is read only when requesting it via the `surface` method.
    2. Data corruption must be detected when opening the file. The
       `surface` method must not fail because of file corruption issues.
    These rules are important to allow smooth operation of the readers in
    the web application `TopoBank`.
    """

    _format = None  # Short internal format string, e.g. 'ce', 'zag', etc.
    _mime_types = None  # MIME type
    _file_extensions = None  # List of common file extensions, without the '.'

    _name = None
    _description = None

    @classmethod
    def format(cls):
        """
        Short string identifier for this file format. Identifier must be
        unique and is typically equal to the file extension of this format.
        """
        if cls._format is None:
            raise RuntimeError('Reader does not provide a format string')
        return cls._format

    @classmethod
    def mime_types(cls):
        """
        MIME types supported by this reader.
        """
        if cls._mime_types is None:
            raise RuntimeError('Reader does not provide MIME types')
        return cls._mime_types

    @classmethod
    def file_extensions(cls):
        """
        A list of typical file extensions for this reader. Can be None if
        there are no typical file extensions.
        """
        if cls._file_extensions is None:
            raise RuntimeError('Reader does not provide file extensions')
        return cls._file_extensions

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
    def nb_surfaces(self):
        """Number of surfaces stored in this container file"""
        return 1

    @abc.abstractmethod
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
        raise NotImplementedError
