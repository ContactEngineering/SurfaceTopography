Usage
=====

The code is documented via Python's documentation strings that can be accesses via the `help` command or by appending a questions mark `?` in ipython/jupyter.

Handling topographies
---------------------

`SurfaceTopography` is designed for handling topography data, either two-dimensional topography maps or line scan. There are three basic topography classes implemented in this module:

- :class:`Topography` is a representation of a two-dimensional topography map that lives on a uniform grid.
- :class:`UniformLineScan` is a representation of a one-dimensional line-scan that lives on a uniform grid.
- :class:`NonuniformLineScan` is a representation of a one-dimensional line-scan that lives on a nonuniform grid. This class assumes that height information in between grid points can be obtained by a linear interpolation.

Nonuniform line-scans are therefore always interpreted as a set of points connected by straight lines
(linear interpolation). No interpolation is carried out for topography maps and uniform line scans.

Topographies can be read from a file through a reader returned by :func:`SurfaceTopography.open_topography`.
Each reader provides an interface to one or more channels in the file.
Each channel returns one of the basic topography classes, depending on the structure of the data contained in the file.
The classes expose a homogeneous interface for handling topographies. Example:

.. code-block:: python

    from SurfaceTopography import open_topography

    # get a handle to the file ("reader")
    reader = open_topography("example.opd")   # you can find this file in the folder 'tests/file_format_examples'

    # each file has a list of channels (one or more)
    print(reader.channels)  # returns list of channels
    ch = reader.channels[0]  # first channel, alternatively use ..
    ch = reader.default_channel  # .. - one of the channels is the "default" channel

    # each channel has some defined meta data
    print(ch.name)  # channel name
    print(ch.physical_sizes)  # lateral dimensions
    print(ch.nb_grid_pts)  # number of grid points
    print(ch.dim)  # number of dimensions (1 or 2)
    print(ch.info)  # more metadata, e.g. 'unit' if unit was given in file

    # you can get a topography from a channel
    topo = ch.topography()   # here meta data from the file is taken

    # each topography has a rich set of methods and properties for meta data and analysis
    print(topo.physical_sizes)  # lateral dimension
    print(topo.rms_height_from_area())  # Root mean square of heights
    h = topo.heights()  # access to the heights array

The raw data can be accesses via the `heights` method that return a one- or two-dimensional array containing height information.
The `positions` method contains return the corresponding positions. For two-dimensional maps, it return two array for the `x` and `y` positions.
For uniform topographies, these positions are uniformly spaced but for nonuniform topographies they may have any value.

Operations on topographies can be analysis functions that compute some value or property,
such as the root mean square height of the topography, or pipeline functions that compute a new topography,
e.g. a detrended one, from the current topography. Both are described in the section :ref:`analysis-functions` below.

Data Orientation
++++++++++++++++

When working with 2D topographies it is useful to know, how the data in `SurfaceTopography` is oriented,
also when compared against the expected image.

After loading a topography, e.g. by

.. code-block:: python

    from SurfaceTopography import open_topography
    reader = open_topography("example.opd")   # you can find this file in the folder 'tests/file_format_examples'
    topo = reader.topography()  # returns the default channel

the heights array can be accessed by

.. code-block:: python

    topo.heights()

or if you need also the coordinates of the heights, use

.. code-block:: python

    topo.positions_and_heights()

If matplotlib has been installed, these heights can be plotted by

.. code-block:: python

    import matplotlib.pyplot as plt
    plt.pcolormesh(topo.heights().T)   # only heights, axes labels are just indices
    # or
    plt.pcolormesh(*topo.positions_and_heights())   # heights and coordinates, axes labels are positions

These two variants plot the origin in the lower left, in a typical cartesian coordinate system.
If you like to have a plot of the topography as seen during measurement, similar to the output
of other software as e.g. Gwyddion, use

.. code-block:: python

   plt.imshow(topo.heights().T)






.. _analysis-functions:

Analysis functions
++++++++++++++++++

All topography classes implement the following analysis functions that can return scalar values or more complex properties. They can be accessed as methods of the topography classes.

- `mean`: Compute the mean value.
- `rms_height_from_area`: Computes the root mean square height of the topography by integrating over the area. (This is the value known as 'Sq'.)
- `rms_height_from_profile`: Computes the root mean square height of the topography as the average of the rms height of individual line scans (profiles) in x-direction. (This is the value known as 'Rq'.)
- `rms_gradient`: Computes the root mean square gradient.
- `rms_slope_from_profile`: Computes the root mean square slope as the average of the rms slope of individual line scans (profiles) in x-direction. Note that there is a factors of sqrt(2) between this values and the rms gradient.
- `rms_curvature_from_area`: Computes the root mean square curvature by integrating over the area.
- `rms_curvature_from_profile`: Computes the root mean square curvature as the average of the rms curvature of individual line scans (profiles) in x-direction.
- `power_spectrum_from_profile`: Computes the one-dimensional power-spectrum (PSD). For two-dimensional topography maps, this functions returns the mean value of all PSDs across the perpendicular direction.
- `power_spectrum_from_area`: Only two-dimensional maps: Computes the radially averaged PSD.
- `autocorrelation_from_profile`: Computes the one-dimensional height difference autocorrelation function (ACF). For two-dimensional topography maps, this functions returns the mean value of all PSDs across the perpendicular direction.
- `autocorrelation_from_area`: Only two-dimensional maps: Computes the radially averaged height difference autocorrelation function.
- `variable_bandwidth_from_profile`: Computes the one-dimentional scan-size dependent rms-height using the variable bandwidth method.
- `variable_bandwidth_from_area`: Computes the two-dimentional scan-size dependent rms-height using the variable bandwidth method.


Example:::

    from SurfaceTopography import read_topography
    topo = read_topography('my_surface.opd')
    print('rms height (Sq) =', topo.rms_height_from_area())
    print('rms gradient =', topo.rms_gradient())
    print('rms curvature =', topo.rms_curvature_from_area())

Pipelines
+++++++++

Pipeline functions return a new topography.
This topography does not own the original data but executes the full pipeline everytime `heights` is executed.
By using the pipeline, this topography is not only a (pseudo-) height container
but also documents the whole process leading from the raw heights to the current heights.
The `squeeze` method returns a new topography that contains the data returned by the pipeline.
Pipelines can be concatenated together.

- `scale`: Rescale all heights by a certain factor.
- `detrend`: Compute a detrended topography.

Example:::

    from SurfaceTopography import read_topography
    topo = read_topography('my_surface.opd')
    print('rms height before detrending =', topo.rms_height_from_area())
    print('rms height after detrending =', topo.detrend(detrend_mode='curvature').rms_height_from_area())
    print('rms height after detrending and rescaling =',
          topo.detrend(detrend_mode='curvature').scale(2.0).rms_height_from_area())
