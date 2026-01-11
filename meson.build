# https://mesonbuild.com/
project(
    'SurfaceTopography', # Project name
    'cpp', # Project type. We need a C++ compiler.
    default_options : ['cpp_std=c++17'], # Yes, we need C++17, at least for constexpr
    version: run_command('python', '-m', 'DiscoverVersion', check: true).stdout().strip(),
)

# https://mesonbuild.com/Python-module.html
pymod = import('python')
python = pymod.find_installation('python3',
    required: true,
)

eigen = dependency('eigen3', version: '>=3.4.0')
pybind11 = dependency('pybind11')

subdir('cpp')
subdir('SurfaceTopography')