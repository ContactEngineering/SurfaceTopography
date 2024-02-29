#
# Copyright 2020-2022 Lars Pastewka
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

"""
Tracing bibliography through function calls.
"""

_default_dois = set()


class doi(object):
    """
    To add bibliographic information, simply decorate the function:

    ```
    @doi('10.1088/2051-672X/aa51f8')
    def power_spectrum(topography):
        ...
    ```

    The DOIs can be requested by passing a `doi` argument with an empty set to
    the function:

    ```
    dois = set()
    power_spectrum(topography, dois=dois)
    print(dois)  # Prints the relevant bibliographic information
    ```

    Note that the dois need to be evaluated directly after the function call.
    Any additional function calls will continue to populate the set with
    bibliography information.
    """

    dois = set()

    _n = 0

    def __init__(self, *args):
        self._add_these_dois = args

    def __call__(self, func):
        def func_with_doi(*args, **kwargs):
            if 'dois' in kwargs:
                # `dois` is present. We store a reference to this set that
                # will be updated in subsequent function calls.
                doi.dois = kwargs['dois']
                # Update this with the default set of DOIs
                doi.dois.update(_default_dois)
                del kwargs['dois']
                doi._n = 0
            doi.dois.update(self._add_these_dois)
            doi._n += 1
            retvals = func(*args, **kwargs)
            doi._n -= 1
            if doi._n == 0:
                # We have reached the point where the original `dois` argument
                # was passed. Create a new set such that subsequent calls don't
                # contaminate the bibliography.
                doi.dois = set()
            return retvals

        return func_with_doi
