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
from .Reader import ContainerMember  # noqa: F401
from .Schema import (  # noqa: F401
    CONTAINER_METADATA_FILENAME,
    LEGACY_METADATA_FILENAME,
    ContainerMeta,
    SurfaceMeta,
    TopographyMeta,
    load_container_metadata,
)
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
    Detect the container format based on the file content.

    Parameters
    ----------
    fobj : str or file-like object
        File name or file-like object.

    Returns
    -------
    format : str
        Short format identifier, e.g. 'ce' or 'zag'.
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


def open_container(fobj, format=None, **kwargs):
    r"""
    Returns a container reader object for the file `fobj`.

    Parameters
    ----------
    fobj : str or filelike object
        Path of the file or file-like object.
    format : str, optional
        Specify in which format the file should be interpreted.
        (Default: None, which means autodetect file format)
    **kwargs : dict
        Additional keyword arguments passed to the reader's constructor
        (e.g. `datafile_keys` or `ignore_filters` of `CEReader`).

    Returns
    -------
    Instance of a :class:`SurfaceTopography.Container.IO.Reader.ContainerReaderBase`
    subclass according to the format.
    """  # noqa: E501
    if not hasattr(fobj, 'read'):  # fobj is a path
        if not os.path.isfile(fobj):
            raise FileNotFoundError("file {} not found".format(fobj))

    if format is None:
        msg = ""
        for reader in readers:
            try:
                return reader(fobj, **kwargs)
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
        return lookup_reader_by_format[format](fobj, **kwargs)


def read_container(fn, format=None, **kwargs):
    r"""
    Returns a surface container object representing a list of topographies.

    Parameters
    ----------
    fn : str or filelike object
        Path of the file or file-like object.
    format : str, optional
        Specify in which format the file should be interpreted.
        (Default: None, which means autodetect file format)
    **kwargs : dict
        Additional keyword arguments passed to the reader's constructor
        (e.g. `datafile_keys` or `ignore_filters` of `CEReader`).

    Returns
    -------
    containers : list of :obj:`SurfaceContainer`
        One container per surface stored in the file. For the formats
        supported here, these are :obj:`LazySurfaceContainer` instances that
        read the individual topographies on access.
    """
    containers = []
    with open_container(fn, format=format, **kwargs) as reader:
        for i in range(reader.nb_containers):
            containers += [reader.container(index=i)]
    return containers


def read_published_container(publication_url, **request_args):
    """
    Download a published container from contact.engineering.

    The publication URL (e.g. `https://contact.engineering/go/<id>/` or the
    DOI URL `https://doi.org/10.57703/ce-<id>`) is
    requested with the header `Accept: application/json`; the JSON response
    contains a `download_url` from which the container ZIP file is
    downloaded and read with `read_container`.

    Parameters
    ----------
    publication_url : str
        Full URL of the publication.
    **request_args
        Additional keyword arguments passed to `requests.get` (for both
        requests). A `headers` dictionary is merged with the `Accept`
        header. (Default timeout: 60 seconds)

    Returns
    -------
    containers : list of :obj:`SurfaceContainer`
        Surface containers read from the downloaded file.
    """
    # Default to a finite timeout so a stale server cannot hang the caller
    # indefinitely; can be overridden through `request_args`
    request_args.setdefault('timeout', 60)

    # If we send json as a request header, then contact.engineering will response with a JSON dictionary
    custom_headers = request_args.pop('headers', {})
    response = requests.get(publication_url,
                            headers={'Accept': 'application/json', **custom_headers},
                            **request_args)
    response.raise_for_status()
    data = response.json()
    download_url = data['download_url']

    # Then download and read container
    container_response = requests.get(download_url, headers=custom_headers, **request_args)
    container_response.raise_for_status()
    container_file = io.BytesIO(container_response.content)
    return read_container(container_file)
