# https://mesonbuild.com/Python-module.html

# Build and install the extension module
# This unified module contains functions from both the former C module
# (_SurfaceTopography) and the former C++ module (_SurfaceTopographyPP)
module = python.extension_module(
    '_SurfaceTopography', # Name of the module (unified)
    [  # List of sources
        # From former C module
        'autocorrelation.cpp',
        'bicubic.cpp',
        'patchfinder.cpp',
        # From former C++ module
        'bearing_area.cpp',
        # Unified bindings
        'module.cpp'
    ],
    install: true,  # Install it
    dependencies: [ # List of dependencies
        python.dependency(), # Add Python.h as dependency
        eigen,
        pybind11
    ]
)
