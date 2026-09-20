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
import dataclasses
from typing import Callable, Optional

from .Schema import TopographyMeta


@dataclasses.dataclass
class ContainerMember:
    """
    A single raw data file (measurement) stored inside a container.

    This is the *ingestion* view of a container: unlike
    :meth:`ContainerReaderBase.container`, which decodes measurements into
    topographies for analysis, a member gives access to the verbatim bytes of
    the underlying data file plus the little metadata the container format
    carries about it. Web applications use this to store the raw file and
    defer decoding.
    """

    #: Human-readable name of the measurement (typically the name of the file
    #: that was originally imported into the container).
    name: str

    #: Callable that returns a fresh binary stream with the verbatim bytes of
    #: the raw data file. Each call returns a new stream; the caller is
    #: responsible for closing it.
    open: Callable

    #: Whether the measurement is visible. Some container formats (e.g.
    #: Keyence ZAG) can mark measurements as hidden without deleting them;
    #: hidden measurements are excluded from `container()` but reported here
    #: so that callers can account for them.
    visible: bool = True

    #: Validated per-measurement metadata, if the container format provides
    #: it (contact.engineering containers do, ZAG does not).
    metadata: Optional[TopographyMeta] = None


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
    def nb_containers(self):
        """Number of surfaces stored in this container file"""
        return 1

    def members(self, index=0):
        """
        Enumerate the raw data files (measurements) stored in this container,
        without decoding them into topographies.

        This is the ingestion companion to :meth:`container`: it exposes the
        verbatim bytes of each measurement's data file plus the metadata the
        container carries about it, so that callers (e.g. web applications)
        can store the raw files and defer decoding. Hidden measurements are
        included, flagged with ``visible=False``; :meth:`container` excludes
        them.

        Arguments
        ---------
        index : int
            Index of the container to enumerate.
            (Default: 0, which enumerates the first container)

        Returns
        -------
        members : list of :obj:`ContainerMember`
            All measurements stored in this container.
        """
        raise NotImplementedError(
            f"The {self.name()} reader does not support member enumeration."
        )

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
