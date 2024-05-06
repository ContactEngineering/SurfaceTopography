# https://mesonbuild.com/Python-module.html

# Build and install the extension module
module = python.extension_module(
    '_SurfaceTopographyPP', # Name of the module
    [  # List of sources
        'bearing_area.cpp',
        'module.cpp'
    ],
    install: true,  # Install it
    dependencies: [ # List of dependencies
        python.dependency(), # Add Python.h as dependency
        eigen,
        pybind11
    ]
)
