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
    print(ch.channel_id)  # stable unique identifier for this channel
    print(ch.is_height_channel)  # True if this is a height/topography channel
    print(ch.physical_sizes)  # lateral dimensions
    print(ch.nb_grid_pts)  # number of grid points
    print(ch.dim)  # number of dimensions (1 or 2)
    print(ch.info)  # more metadata, e.g. 'unit' if unit was given in file

    # you can get a topography from a channel - multiple selection methods:
    topo = ch.topography()  # from channel object
    topo = reader.topography(channel_index=0)  # by index
    topo = reader.topography(channel_id="Height")  # by stable ID (recommended for databases)
    topo = reader.topography(height_channel_index=0)  # by height-only index (backwards compatible)

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
- `median`: Compute the median value.
- `min`: Compute the minimum value.
- `max`: Compute the maximum value.
- `bandwidth`: Compute the bandwidth of the topography (minimum and maximum lateral length scales).
- `rms_height_from_area` or `Sq`: Computes the root mean square height of the topography by integrating over the area. (This is the value known as 'Sq'.)
- `rms_height_from_profile` or `Rq`: Computes the root mean square height of the topography as the average of the rms height of individual line scans (profiles) in x-direction. (This is the value known as 'Rq'.)
- `moment`: Computes an arbitrary moment of the heights.
- `mad_height`: Computes the mean absolute deviation of the height.
- `rms_gradient`: Computes the root mean square gradient.
- `rms_slope_from_profile` or `Rdq`: Computes the root mean square slope as the average of the rms slope of individual line scans (profiles) in x-direction. Note that there is a factors of sqrt(2) between this values and the rms gradient.
- `rms_curvature_from_area`: Computes the root mean square curvature by integrating over the area.
- `rms_curvature_from_profile` or `Rddq`: Computes the root mean square curvature as the average of the rms curvature of individual line scans (profiles) in x-direction.
- `power_spectrum_from_profile`: Computes the one-dimensional power-spectrum (PSD). For two-dimensional topography maps, this functions returns the mean value of all PSDs across the perpendicular direction.
- `power_spectrum_from_area`: Only two-dimensional maps: Computes the radially averaged PSD.
- `autocorrelation_from_profile`: Computes the one-dimensional height difference autocorrelation function (ACF). For two-dimensional topography maps, this functions returns the mean value of all PSDs across the perpendicular direction.
- `autocorrelation_from_area`: Only two-dimensional maps: Computes the radially averaged height difference autocorrelation function.
- `variable_bandwidth_from_profile`: Computes the one-dimentional scan-size dependent rms-height using the variable bandwidth method.
- `variable_bandwidth_from_area`: Computes the two-dimentional scan-size dependent rms-height using the variable bandwidth method.
- `derivative`: Computes the (scale-dependent) derivative of the topography. This is described in detail in the paper `Scale-dependent roughness parameters for topography analysis`_.
- `scale_dependent_statistical_property`: Computes the scale-dependent statistical property (SDRP) of the topography. This is described in detail in the paper `Scale-dependent roughness parameters for topography analysis`_.
- `scale_dependent_slope_from_area`: Computes the scale-dependent slope from the area. This is a convenience function yielding a specific SDRP.
- `scale_dependent_slope_from_profile`: Computes the scale-dependent slope from the profile. This is a convenience function yielding a specific SDRP.
- `scale_dependent_curvature_from_area`: Computes the scale-dependent curvature from the area. This is a convenience function yielding a specific SDRP.
- `scale_dependent_curvature_from_profile`: Computes the scale-dependent curvature from the profile. This is a convenience function yielding a specific SDRP.
- `scaning_proper_reliability_cutoff`: Use the SDRP to estimate a resolution cutoff for the measurement given the tip size of the probe. (The theory behind this analysis is described in detail in `Scale-dependent roughness parameters for topography analysis`_.)

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

**Data correction:**

- `detrend`: Compute a detrended topography by removing polynomial trends (constant, linear, quadratic, etc.).
- `scan_line_align`: Remove scan line artifacts from AFM data by fitting and subtracting per-line polynomials and aligning adjacent lines. Supports variable polynomial degree for scanner bow correction.
- `fill_undefined_data`: Fill undefined/missing data points using harmonic interpolation.
- `interpolate_undefined_data`: Interpolate undefined data points.

**Scaling and conversion:**

- `scale`: Rescale all heights by a certain factor.
- `to_unit`: Convert the topography to a different unit (scaling lateral lengths and heights).
- `to_nonuniform`: Convert a uniform topography to a nonuniform representation.

**Filtering:**

- `window`: Apply a windowing function to the topography (for spectral analysis).
- `filter`: Apply a Fourier filter to the topography.
- `shortcut`: Apply a short-wavelength cutoff filter (high-pass).
- `longcut`: Apply a long-wavelength cutoff filter (low-pass).

**Interpolation:**

- `interpolate_fourier`: Interpolate the topography using Fourier methods.
- `interpolate_linear`: Linear interpolation at arbitrary positions (2D maps only).
- `interpolate_bicubic`: Bicubic interpolation at arbitrary positions (2D maps only).
- `mirror_stitch`: Create a periodic topography by mirror stitching (2D maps only).

Example::

    from SurfaceTopography import read_topography
    topo = read_topography('my_surface.opd')
    print('rms height before detrending =', topo.rms_height_from_area())
    print('rms height after detrending =', topo.detrend(detrend_mode='curvature').rms_height_from_area())
    print('rms height after detrending and rescaling =',
          topo.detrend(detrend_mode='curvature').scale(2.0).rms_height_from_area())

Scan line alignment example (for AFM data with stripe artifacts)::

    from SurfaceTopography import read_topography
    topo = read_topography('afm_scan.ibw')

    # Remove per-line tilt and align adjacent lines
    aligned = topo.scan_line_align()

    # For scanner bow correction, use higher polynomial degree
    aligned = topo.scan_line_align(degree=2)  # Removes quadratic curvature per line

    # Can be combined with other pipeline operations
    corrected = topo.scan_line_align().detrend()

Output functions
++++++++++++++++

Output functions are used to save the topography data to a file.

- `to_matrix`: Write a text representation of the height data to file.
- `to_netcdf`: Write the topography to a NetCDF file.
- `to_dzi`: Write the topography as a Deep Zoom Image (DZI) file.
- `to_x3p`: Write the topography to an X3P file (ISO 5436-2 format).
- `to_gwy`: Write the topography to a Gwyddion file.

.. _`Scale-dependent roughness parameters for topography analysis`: https://doi.org/10.1016/j.apsadv.2021.100190