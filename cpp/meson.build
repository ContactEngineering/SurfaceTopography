# https://mesonbuild.com/Python-module.html

eigen3_incdir = include_directories('../external/eigen3')
pybind11_incdir = include_directories('../external/pybind11/include')

# Build and install the extension module
module = python.extension_module(
    '_SurfaceTopographyPP', # Name of the module
    [  # List of sources
        'bearing_area.cpp',
        'module.cpp'
    ],
    install: true,  # Install it
    include_directories: [eigen3_incdir, pybind11_incdir],
    dependencies: [ # List of dependencies
        python.dependency() # Add Python.h as dependency
    ]
)
