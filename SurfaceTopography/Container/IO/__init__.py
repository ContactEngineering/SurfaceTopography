#
# Copyright 2019-2024 Lars Pastewka
#           2022 Johannes Hörmann
#           2020-2021 Michael Röttger
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

import io
import os

import requests

from ...Exceptions import CannotDetectFileFormat, UnknownFileFormat
from .CE import CEReader, write_containers  # noqa: F401
from .ZAG import ZAGReader

readers = [
    CEReader,
    ZAGReader
]

lookup_reader_by_format = {}
for reader in readers:
    lookup_reader_by_format[reader.format()] = reader


def detect_format(fobj):
    """
    Detect file format based on its content.

    Arguments
    ---------
    fobj : filename or file object
    """
    msg = ""
    for reader in readers:
        try:
            reader(fobj)
            return reader.format()
        except Exception as err:
            msg += "tried {}: \n {}\n\n".format(reader.__name__, err)
        finally:
            if hasattr(fobj, 'seek'):
                # if the reader crashes the cursor in the file-like object
                # has to be set back to the top of the file
                fobj.seek(0)
    raise CannotDetectFileFormat(msg)


def open_container(fobj, format=None):
    r"""
    Returns a container reader object for the file `fobj`.

    Parameters
    ----------
    fobj : str or filelike object
        Path of the file or file-like object.
    format : str, optional
        Specify in which format the file should be interpreted.
        (Default: None, which means autodetect file format)

    Returns
    -------
    Instance of a :class:`SurfaceTopography.Container.IO.Reader.ContainerReaderBase`
    subclass according to the format.
    """  # noqa: E501
    if not hasattr(fobj, 'read'):  # fobj is a path
        if not os.path.isfile(fobj):
            raise FileExistsError("file {} not found".format(fobj))

    if format is None:
        msg = ""
        for reader in readers:
            try:
                return reader(fobj)
            except Exception as err:
                msg += "tried {}: \n {}\n\n".format(reader.__name__, err)
            finally:
                if hasattr(fobj, 'seek'):
                    # if the reader crashes the cursor in the file-like object
                    # has to be set back to the top of the file
                    fobj.seek(0)
        raise CannotDetectFileFormat(msg)
    else:
        if format not in lookup_reader_by_format.keys():
            raise UnknownFileFormat(
                "{} not in registered container file formats {}".format(
                    fobj, lookup_reader_by_format.keys()))
        return lookup_reader_by_format[format](fobj)


def read_container(fn, format=None, **kwargs):
    r"""
    Returns a surface container object representing a list of topographies.

    Parameters
    ----------
    fobj : str or filelike object
        Path of the file or file-like object.

    Returns
    -------
    container : list of subclasses of :obj:`SurfaceContainer`
        A list of container objects.
    """
    containers = []
    with open_container(fn, format=format) as reader:
        for i in range(reader.nb_surfaces):
            containers += [reader.container(index=i, **kwargs)]
    return containers


def read_published_container(publication_url, **request_args):
    """
    Download a container from a URL.

    Parameters
    ----------
    publication_url : str
        Full URL of container location.
    request_args : dict
        Additional dictionary passed to `request.get`.

    Returns
    -------
    container : SurfaceContainer
        Surface container object read from URL.
    """
    # If we send json as a request header, then contact.engineering will response with a JSON dictionary
    response = requests.get(publication_url, headers={'Accept': 'application/json'})
    data = response.json()
    download_url = data['download_url']

    # Then download and read container
    container_response = requests.get(download_url, **request_args)
    container_file = io.BytesIO(container_response.content)
    return read_container(container_file)
