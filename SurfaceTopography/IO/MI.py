#
# Copyright 2019-2021 Lars Pastewka
#           2020-2021 Michael Röttger
#           2019-2020 Kai Haase
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

import numpy as np

from ..UniformLineScanAndTopography import Topography
from ..UnitConversion import mangle_length_unit_utf8

from .common import OpenFromAny
from .Reader import ReaderBase, CorruptFile, ChannelInfo, MetadataAlreadyFixedByFile

image_head = b'fileType      Image\n'
spec_head = b'fileType      Spectroscopy\n'

magic_data = b'data          \n'
magic_data_binary = b'data          BINARY\n'
magic_data_binary32 = b'data          BINARY_32\n'
magic_data_ascii = b'data          ASCII'


class MIReader(ReaderBase):
    _format = 'mi'
    _name = 'Molecular imaging data file'
    _description = '''
This reader opens Agilent Technologies (Molecular Imaging) AFM files saved
in the MI format. This format contains information on the physical size of the
topography map as well as its units.
'''

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self.file_path = file_path

        # Start of the data in the file
        self.data_start = None
        # If image oder spectroscopy
        self.is_image = None
        # Format of the saved data
        self.data_type = None

        self.mifile = None

        # Process the header and read in metadata
        self.process_header()

    def process_header(self):
        with OpenFromAny(self.file_path, 'rb') as f:
            self.lines = f.readlines()

            # Find out if image or spectroscopy
            if self.lines[0] == image_head:
                self.is_image = True
            elif self.lines[0] == spec_head:
                self.is_image = False
            else:
                raise CorruptFile

            # Find the start of the height data and denote data type
            if magic_data in self.lines:
                header_size = self.lines.index(magic_data)
                self.data_type = 'text'
            elif magic_data_binary in self.lines:
                header_size = self.lines.index(magic_data_binary)
                self.data_type = 'binary'
            elif magic_data_binary32 in self.lines:
                header_size = self.lines.index(magic_data_binary32)
                self.data_type = 'binary32'
            elif magic_data_ascii in self.lines:
                header_size = self.lines.index(magic_data_ascii)
                self.data_type = 'ascii'
            else:
                raise CorruptFile

            # Save start of data for later reading of the matrix
            self.data_start = header_size + 1

            # Create mifile from header, reading out meta data and channel info
            if self.is_image:
                self.mifile = read_header_image(self.lines[1:header_size])
            else:  # TODO
                self.mifile = read_header_spect(self.lines[1:header_size])

            # Reformat the metadata
            for buf in self.mifile.channels:
                buf.meta['name'] = buf.name
                buf.unit = mangle_length_unit_utf8(buf.meta.pop('bufferUnit'))
                buf.meta['range'] = buf.meta.pop('bufferRange')
                buf.meta['label'] = buf.meta.pop('bufferLabel')

            self._physical_sizes = float(self.mifile.meta['xLength']), float(self.mifile.meta['yLength'])
            self._nb_grid_pts = int(self.mifile.meta['xPixels']), int(self.mifile.meta['yPixels'])

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={}, periodic=False,
                   subdomain_locations=None, nb_subdomain_grid_pts=None):
        if channel_index is None:
            channel_index = self._default_channel_index

        if subdomain_locations is not None or \
                nb_subdomain_grid_pts is not None:
            raise RuntimeError(
                'This reader does not support MPI parallelization.')

        output_channel = self.mifile.channels[channel_index]

        buffer = b''.join(self.lines[self.data_start:])
        buffer = [chr(_) for _ in buffer]

        # Read height data
        if self.is_image:
            if self.data_type == 'binary':
                dt = "i2"
                encode_length = 2
                type_range = 32768
            elif self.data_type == 'binary32':
                dt = "i4"
                encode_length = 4
                type_range = 2147483648
            else:  # text or ascii
                type_range = 32768

            start = int(self.mifile.xres) * int(
                self.mifile.yres) * encode_length * channel_index
            end = int(self.mifile.xres) * int(
                self.mifile.yres) * encode_length * (channel_index + 1)

            data = ''.join(buffer[start:end])
            out = np.frombuffer(str.encode(data, "raw_unicode_escape"), dt)
            out = out.reshape((int(self.mifile.xres), int(self.mifile.yres)))

            # Undo normalizing with range of data type
            out = out / type_range

            # If scan direction is upwards, flip the height map vertically
            if self.mifile.meta['scanUp']:
                out = np.flip(out, 0)

            # Multiply the heights with the bufferRange
            out *= float(output_channel.meta['range'])
        else:
            pass  # TODO

        joined_meta = {**self.mifile.meta, **output_channel.meta}
        info = info.copy()
        info.update(joined_meta)

        if unit is not None:
            raise MetadataAlreadyFixedByFile('unit')

        # Initialize heights with transposed array in order to match Gwdyydion
        # when plotted with pcolormesh(t.heights().T), except that the y axis
        # is flipped because the origin is in lower left with pcolormesh;
        # imshow(t.heights().T) shows the image like gwyddion
        t = Topography(heights=out.T,
                       physical_sizes=self._check_physical_sizes(
                           physical_sizes, self._physical_sizes),
                       unit=output_channel.unit, info=info, periodic=periodic)
        if height_scale_factor is not None:
            t = t.scale(height_scale_factor)
        return t

    @property
    def channels(self):
        return [ChannelInfo(self, i, name=channel.meta['name'],
                            dim=len(self._nb_grid_pts),
                            nb_grid_pts=self._nb_grid_pts,
                            physical_sizes=self._physical_sizes,
                            unit=channel.unit,
                            info=channel.meta)
                for i, channel in enumerate(self.mifile.channels)]

    @property
    def info(self):
        """ Return all the available metadata as a dict. """
        return self.mifile.meta

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__


def read_header_image(header):
    """
    Reads in global metadata and information about about included channels as
    well as their metadata.
    :param header: The header as a line of text.
    :return:
    MIFile item containing the metadata, with the channels as a list of
    'buffers'.
    """
    # This object will store the file-wide metadata
    mifile = MIFile()

    # False while reading global metadata, gets true if we start to read in the
    # channel info
    reading_buffers = False

    for line in header:
        line = line.decode("utf-8")

        # As soon as we see a line starting with 'bufferLabel', we know we are
        # now reading in channels
        if line.startswith('bufferLabel'):
            # Create a new channel with the id as name
            channel = Channel(name=str.strip(line[14:]))
            mifile.channels.append(channel)
            reading_buffers = True

        if line[0] == ' ':
            continue

        # For all key value pairs in the file:
        # Append to global metadata oder channel metadata, depending on our
        # state
        key = str.strip(line[:14])
        value = str.strip(line[14:])

        if reading_buffers:
            mifile.channels[-1].meta[key] = value
        else:
            mifile.meta[key] = value

        # Catch 'special' metadata
        if key == "xPixels":
            mifile.xres = value
        elif key == "yPixels":
            mifile.yres = value
    return mifile


def read_header_spect(header):
    """
    Reads in metadata out of the header.
    :param header: The header.
    :return:
    MIFile item containing the metadata.
    """
    pass


class Channel:
    """
    Class structure for a channel contained in the file.
    Has a name and metadata (height data is not needed since it is returned
    directly).
    """

    def __init__(self, name=None, unit=None, meta=None):
        if meta is None:
            meta = dict()

        self.name = name
        self.unit = unit
        self.meta = meta


class MIFile:
    """
    Class structure for the while file. Has a list of channels, global metadata
    and a nb_grid_pts.
    """

    def __init__(self, res=(0, 0), channels=None, meta=None):
        if channels is None:
            channels = list()
        if meta is None:
            meta = dict()

        self.xres = res[0]
        self.yres = res[1]
        self.channels = channels
        self.meta = meta
