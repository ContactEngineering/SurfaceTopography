#
# Copyright 2019 Antoine Sanner
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

import numpy as np

from PyCo.Topography.IO.Reader import ReaderBase, CorruptFile
from PyCo.Topography import Topography


image_head = b'fileType      Image\n'
spec_head = b'fileType      Spectroscopy\n'

magic_data = b'data          \n'
magic_data_binary = b'data          BINARY\n'
magic_data_binary32 = b'data          BINARY_32\n'
magic_data_ascii = b'data          ASCII'


class MIReader(ReaderBase):

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path, size=None, info=None):
        super().__init__(size, info)

        self.file_path = file_path

        # Start of the data in the file
        self.data_start = None
        # If image oder spectroscopy
        self.is_image = None
        # Format of the saved data
        self.data_type = None
        # The default channel
        self._default_channel = None

        self.mifile = None

        # Process the header and read in metadata
        self.process_header()

    def process_header(self):
        with open(self.file_path, "rb") as f:
            lines = f.readlines()

            # Find out if image or spectroscopy
            if lines[0] == image_head:
                self.is_image = True
            elif lines[0] == spec_head:
                self.is_image = False
            else:
                raise CorruptFile

            # Find the start of the height data and denote data type
            if magic_data in lines:
                header_size = lines.index(magic_data)
                self.data_type = 'text'
            elif magic_data_binary in lines:
                header_size = lines.index(magic_data_binary)
                self.data_type = 'binary'
            elif magic_data_binary32 in lines:
                header_size = lines.index(magic_data_binary32)
                self.data_type = 'binary32'
            elif magic_data_ascii in lines:
                header_size = lines.index(magic_data_ascii)
                self.data_type = 'ascii'
            else:
                raise CorruptFile

            # Save start of data for later reading of the matrix
            self.data_start = header_size + 1

            # Create mifile from header, reading out meta data and channel info
            if self.is_image:
                self.mifile = read_header_image(lines[1:header_size])
            else:  # TODO
                self.mifile = read_header_spect(lines[1:header_size])

            # Reformat the metadata
            for buf in self.mifile.channels:
                buf.meta['name'] = buf.name
                buf.meta['unit'] = buf.meta.pop('bufferUnit')
                buf.meta['range'] = buf.meta.pop('bufferRange')
                buf.meta['label'] = buf.meta.pop('bufferLabel')

            self._size = float(self.mifile.meta['xLength']), \
                float(self.mifile.meta['yLength'])
            self._resolution = int(self.mifile.meta['xPixels']), \
                int(self.mifile.meta['yPixels'])

            self._default_channel = 0  # Maybe search for id 'Topography' in the future

    def topography(self, size=None, channel=None):
        if channel is None:
            channel = self._default_channel

        with open(self.file_path, "rb") as f:
            lines = f.readlines()

            output_channel = self.mifile.channels[channel]

            buffer = b''.join(lines[self.data_start:])
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

                start = int(self.mifile.xres) * int(self.mifile.yres) * encode_length * channel
                end = int(self.mifile.xres) * int(self.mifile.yres) * encode_length * (channel + 1)

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
        return Topography(heights=out, size=self._size, info=joined_meta)

    def channels(self):
        """ Get a list of available channels. """
        return [channel.meta for channel in self.mifile.channels]

    def info(self):
        """ Return all the available metadata as a dict. """
        return self.mifile.meta


def read_header_image(header):
    """
    Reads in global metadata and information about about included channels asw ell as their metadata.
    :param header: The header as a line of text.
    :return:
    MIFile item containing the metadata, with the channels as a list of 'buffers'.
    """
    # This object will store the file-wide metadata
    mifile = MIFile()

    # False while reading global metadata, gets true if we start to read in the channel info
    reading_buffers = False

    for line in header:
        line = line.decode("utf-8")

        # As soon as we see a line starting with 'bufferLabel', we know we are now reading in channels
        if line.startswith('bufferLabel'):
            # Create a new channel with the id as name
            channel = Channel(name=str.strip(line[14:]))
            mifile.channels.append(channel)
            reading_buffers = True

        if line[0] == ' ':
            continue

        # For all key value pairs in the file:
        # Append to global metadata oder channel metadata, depending on our state
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
    Has a name and metadata (height data is not needed since it is returned directly).
    """

    def __init__(self, name=None, meta=None):
        if meta is None:
            meta = dict()

        self.name = name
        self.meta = meta


class MIFile:
    """
    Class structure for the while file. Has a list of channels, global metadata and a resolution.
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
