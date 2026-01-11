#
# Copyright 2023-2025 Lars Pastewka
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
Block definition class for TLV-based file formats (FRT, MNT).

This module is intentionally kept free of dependencies on other
SurfaceTopography modules to avoid circular import issues.
"""


class BlockDefinition:
    """
    Declarative definition for a TLV block structure.

    Parameters
    ----------
    fields : list of tuples, optional
        Field definitions for decode(): [(name, format), ...]
        Format codes follow struct module plus extensions in binary.py.
    text : bool, optional
        If True, entire block content is ASCII text. Default: False.
    container : bool or dict, optional
        If True, block contains nested TLV entries (parsed generically).
        If dict, maps child tag IDs to BlockDefinition instances for
        structured parsing of nested content. Default: False.
    trailing_data : bool, optional
        If True, record file offset of data following the defined fields.
        Useful for blocks with variable-length data at the end.
        Default: False.
    subblocks : tuple, optional
        (count_field, field_list) for repeated sub-structures.
        count_field is the name of a field containing the repeat count.
        field_list is the structure definition for each subblock.
        Default: None.
    skip_rest : bool, optional
        If True, skip any remaining bytes after parsing fields.
        Useful for blocks with padding or unknown trailing data.
        Default: False.

    Examples
    --------
    Simple structured block:

    >>> BlockDefinition(fields=[
    ...     ('width', 'I'),
    ...     ('height', 'I'),
    ... ])

    Text block:

    >>> BlockDefinition(text=True)

    Block with trailing data:

    >>> BlockDefinition(
    ...     fields=[('count', 'I'), ('flags', 'I')],
    ...     trailing_data=True
    ... )

    Block with repeated subblocks:

    >>> BlockDefinition(
    ...     fields=[('nb_items', 'I')],
    ...     subblocks=('nb_items', [('value', 'd'), ('name', '32s')])
    ... )

    Container with known child structures:

    >>> BlockDefinition(container={
    ...     0x0001: BlockDefinition(fields=[('value', 'I')]),
    ...     0x0002: BlockDefinition(text=True),
    ... })
    """

    def __init__(self, fields=None, text=False, container=False,
                 trailing_data=False, subblocks=None, skip_rest=False):
        self.fields = fields
        self.text = text
        self.container = container
        self.trailing_data = trailing_data
        self.subblocks = subblocks
        self.skip_rest = skip_rest

    def __repr__(self):
        parts = []
        if self.fields:
            parts.append(f'fields={self.fields!r}')
        if self.text:
            parts.append('text=True')
        if self.container:
            parts.append(f'container={self.container!r}')
        if self.trailing_data:
            parts.append('trailing_data=True')
        if self.subblocks:
            parts.append(f'subblocks={self.subblocks!r}')
        if self.skip_rest:
            parts.append('skip_rest=True')
        return f'BlockDefinition({", ".join(parts)})'


def block(fields=None, text=False, container=False,
          trailing_data=False, subblocks=None, skip_rest=False):
    """
    Convenience function to create a BlockDefinition.

    See BlockDefinition for parameter documentation.
    """
    return BlockDefinition(
        fields=fields,
        text=text,
        container=container,
        trailing_data=trailing_data,
        subblocks=subblocks,
        skip_rest=skip_rest
    )
