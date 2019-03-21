import warnings
import abc

class ReaderBase(metaclass=abc.ABCMeta):
    def __init__(self, size=None,  info={}):
        self._size = size
        self._info = info

    @property
    def resolution(self):
        return self._resolution

    @property
    def size(self):
        return self._size

    @property
    def info(self):
        return self._info

    def _process_size(self, size):
        if self.size is None:
            if size is None:
                raise ValueError("size could not be extracted from file, you should provide it")
        else:
            if size is None:
                size = self.size
            elif [s == ss for s, ss in zip(size, self.size)] != [True, True]: # both sizes are defined
                warnings.warn("a size different from the specified value"
                              "was present in the file ({})."
                              "we will use the specified value".format(self.size, size))
        return size

    def _process_info(self, info):
        newinfo = self.info.copy()
        newinfo.update(info)
        return newinfo

    @abc.abstractmethod
    def topography(self):
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
    Raised when no reader is able to read the file
    """

class FileFormatMismatch(ReadFileError):
    """
    Raised when the reader cannot interpret the file at all
    (obvious for txt vs binary, but holds also for a header)
    """
    pass

class CorruptFile(ReadFileError):
    """
    Raised when the reader identifies the file format as matching, but there is a mistake, for example the number of points doesn't match
    """
    pass
