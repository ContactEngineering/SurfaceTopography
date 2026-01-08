# https://mesonbuild.com/
project(
    'SurfaceTopography', # Project name
    'c', 'cpp', # Project type. We need a C and C++ compiler.
    default_options : ['cpp_std=c++17'], # Yes, we need C++17, at least for constexpr
    # Version is determined dynamically by meson-python via setuptools_scm.
    # Using a placeholder here; meson-python overrides this via native file.
    version: '0.0.0',
)

# https://mesonbuild.com/Python-module.html
pymod = import('python')
python = pymod.find_installation('python3',
    required: true,
)

eigen = dependency('eigen3', version: '>=3.4.0')
pybind11 = dependency('pybind11')

subdir('c')
subdir('cpp')
subdir('SurfaceTopography')