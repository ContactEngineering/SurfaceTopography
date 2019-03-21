from PyCo.Topography.IO.Reader import ReaderBase
from PyCo.Topography import Topography

class H5Reader(ReaderBase):
    def __init__(self, fobj):

        import h5py
        self._h5 = h5py.File(fobj)

        super().__init__()

    def topography(self, size = None, info={}):
        size =self._process_size(size)
        return Topography(self._h5['surface'][...], size, info=self._process_info(info))