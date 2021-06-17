"""
Tests of the pipeline for topography containers
"""

import os
import pytest

import numpy as np

from SurfaceTopography import read_published_container

def test_scale_dependent_statistical_property():
    c, = read_published_container('https://contact.engineering/go/867nv/')
    s = c.scale_dependent_statistical_property(lambda x, y: np.var(x), n=1, distance=[0.01, 0.1, 1.0, 10], unit='um')
    assert (np.diff(s) < 0).all()
