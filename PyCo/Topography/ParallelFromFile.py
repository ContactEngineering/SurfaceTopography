from  PyCo.Tools import MPIFileIO
from PyCo.Topography import UniformNumpyTopography


# TODO: langfristig sollten alle ladefunktionen nur einmal implementiert werden, sowohl MPI und seriell-fähig

# TODO: Die Funktionen können in der Datei zuerst die Metadaten abtasten.
#       Das braucht man zuerst um die resolution zu bestimmen.
#       Dann erstellt man das substrate entsprechend
#       Erst dann kann man die Topography parallel erstellen


def read_npy(self, fn, substrate):
    """
    Reads the data with the subdomain_decomposition acording to the substrate from npy

    Parameters
    ----------
    self
    fn
    substrate

    Returns
    -------
    ParallelNumpyTopography
    """

    # check that the resolution in the file matches the resolution in the substrate.

    # load relevant data and but it in the ParallelUniformNumpyTopography

    raise NotImplementedError


class MPITopographyLoader():
    def __init__(self, fn, comm, format = None):
        self.size = None  # will stay None if the file doesn't provide the information.
        self.unit = None

        self.mpi_file= MPIFileIO.MPIFileViewFactory(fn, comm, format=format)
        self.dtype = self.mpi_file.dtype
        self.resolution = self.mpi_file.resolution

        # TODO: maybe implement extras specific to Topography , like loading the units and the size

    def getTopography(self, substrate):
        # TODO: Are sometimes the Units Stored?
        return UniformNumpyTopography(
            profile=self.mpi_file.read(subdomain_location=substrate.topography_subdomain_location,
                                       subdomain_resolution=substrate.topography_subdomain_resolution),
                                       subdomain_location=substrate.topography_subdomain_location,
                                       resolution=substrate.resolution, pnp=substrate.pnp
                                    )


# TODO: Does this belong here ?
def save_npy(fn, topography):
    MPIFileIO.save_npy(fn=fn, data=topography.array(), subdomain_location=topography.subdomain_location,
                       resolution=topography.subdomain_resolution, comm=topography.comm)
