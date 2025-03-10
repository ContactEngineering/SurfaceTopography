#
# Copyright 2020-2021 Lars Pastewka
#           2021 Michael Röttger
#           2019-2020 Antoine Sanner
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import warnings
from functools import wraps


def deprecated(version=None, alternative=None):
    """Emit a deprecation warning for a function."""

    def deprecated_decorator(func):

        def _get_warn_str(version=version, alternative=alternative):
            if version is None:
                warn_str = f"Function/property `{func.__name__}` is deprecated."
            else:
                warn_str = f"Function/property `{func.__name__}` was deprecated in SurfaceTopography {version}."
            if alternative is not None:
                warn_str += f" The function/property `{alternative}` provides this functionality now."
            return warn_str

        @wraps(func)
        def deprecated_func(*args, **kwargs):
            warnings.warn(_get_warn_str(version, alternative), DeprecationWarning)
            return func(*args, **kwargs)

        docstring = deprecated_func.__doc__ or ""
        docstring += "\n\n" + _get_warn_str(version, alternative)
        deprecated_func.__doc__ = docstring

        return deprecated_func

    return deprecated_decorator
