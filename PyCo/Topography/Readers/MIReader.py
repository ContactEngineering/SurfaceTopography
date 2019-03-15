import numpy as np

from PyCo.Topography.ParallelFromFile import TopographyLoader
from PyCo.Topography import Topography


class TopographyLoaderMI(TopographyLoader):

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path, size=None, unit=None, info=None):
        super().__init__(size, info)

        with open(file_path, "rb") as f:

            pass

    def topography(self):
        pass

        # return Topography(heights=data, size=size, info=info)
