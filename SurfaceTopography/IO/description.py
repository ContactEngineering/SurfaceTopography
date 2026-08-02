#
# Copyright 2026 Lars Pastewka
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
Serialization of declarative file layouts to and from the JSON documents
defined by the format description contract
(`docs/format_description_contract.rst`).

`layout_to_dict` turns a layout tree (built from the classes in `binary.py`
and `Reader.py`) into a JSON-compatible dictionary; `layout_from_dict`
rehydrates it. Hooks that are Python callables rather than expressions
(see `SurfaceTopography.IO.expr`) serialize to *opaque* markers — a
document containing opaque markers is incomplete and cannot be rehydrated
or executed by a foreign-language engine. `count_opaque` reports them.
"""

import numpy as np

from ..Exceptions import (
    CorruptFile,
    FileFormatMismatch,
    UnsupportedFormatFeature,
    UnsupportedSchema,
)
from . import expr
from .binary import (
    BinaryArray,
    BinaryStructure,
    Convert,
    RawBuffer,
    TextBuffer,
    TextHeader,
    TextLine,
    TextMatrix,
    TLVContainer,
    Validate,
    ValidationError,
    XMLStructure,
    ZipContainer,
    ZlibBlockChain,
    _identity,
)
from .Reader import (
    Check,
    CompoundLayout,
    DeclarativeReaderBase,
    For,
    If,
    Seek,
    SizedChunk,
    Skip,
    Switch,
    While,
)

SCHEMA_VERSION = 1

# Error taxonomy names, per the contract
_ERROR_NAMES = {
    FileFormatMismatch: "format_mismatch",
    CorruptFile: "corrupt_file",
    UnsupportedFormatFeature: "unsupported_feature",
    # A plain validation failure means the file is corrupt
    ValidationError: "corrupt_file",
}
_ERROR_CLASSES = {
    "format_mismatch": FileFormatMismatch,
    "corrupt_file": CorruptFile,
    "unsupported_feature": UnsupportedFormatFeature,
}

# Unset hooks default to the shared `_identity` function and are detected
# (and omitted) by identity with it


def _opaque(value):
    """Marker for a hook that is not serializable (a Python callable)."""
    return {"kind": "opaque", "repr": repr(value)}


def encode_value(value):
    """
    Encode a hook or parameter value: expressions to their AST, literals to
    plain JSON, remaining callables to opaque markers.
    """
    if isinstance(value, expr.Expr):
        return value.to_dict()
    elif isinstance(value, np.dtype):
        # A fixed dtype is the expression `F.dtype("<i4")`
        return expr.F.dtype(value.str).to_dict()
    elif isinstance(value, bytes):
        return expr.BytesLit(value).to_dict()
    elif callable(value):
        return _opaque(value)
    elif isinstance(value, (tuple, list)):
        return [encode_value(v) for v in value]
    elif isinstance(value, dict):
        return {key: encode_value(v) for key, v in value.items()}
    elif isinstance(value, (bool, int, float, str)) or value is None:
        return value
    else:
        return _opaque(value)


def decode_value(value):
    """Inverse of `encode_value`."""
    if isinstance(value, dict):
        if "kind" in value:
            if value["kind"] == "opaque":
                raise UnsupportedSchema(
                    f"Cannot rehydrate opaque hook {value.get('repr')}; the "
                    f"document is incomplete."
                )
            return expr.from_dict(value)
        return {key: decode_value(v) for key, v in value.items()}
    elif isinstance(value, list):
        return [decode_value(v) for v in value]
    else:
        return value


def _encode_error(exception_class):
    name = _ERROR_NAMES.get(exception_class)
    if name is None:
        return _opaque(exception_class)
    return name


def _decode_error(name):
    try:
        return _ERROR_CLASSES[name]
    except KeyError:
        raise UnsupportedSchema(f"Unknown error taxonomy name `{name}`.")


def _encode_field_hooks(hooks):
    """Encode the validator/converter hooks of a `BinaryStructure` field."""
    encoded = []
    for hook in hooks:
        if isinstance(hook, Validate):
            encoded.append(
                {
                    "validate": {
                        "value": encode_value(hook._value),
                        "error": _encode_error(hook._exception),
                    }
                }
            )
        elif isinstance(hook, Convert):
            encoded.append({"convert": encode_value(hook._fun)})
        elif isinstance(hook, expr.Expr):
            encoded.append({"convert": hook.to_dict()})
        else:
            encoded.append({"convert": _opaque(hook)})
    return encoded


def _decode_field_hooks(encoded):
    hooks = []
    for hook in encoded:
        if "validate" in hook:
            hooks.append(
                Validate(
                    decode_value(hook["validate"]["value"]),
                    _decode_error(hook["validate"]["error"]),
                )
            )
        elif "convert" in hook:
            hooks.append(decode_value(hook["convert"]))
        else:
            raise UnsupportedSchema(f"Unknown field hook {hook!r}.")
    return hooks


###
# Per-class encoders. Each returns the contract-style dictionary for one
# layout node.
###


def _encode_name(layout):
    return encode_value(layout._name)


# Format codes whose size differs between native (`@`) and standard (`=`)
# sizes, or that are unavailable with standard sizes altogether. `@` can
# only be rewritten to `=` for structures that avoid them.
_NATIVE_SIZE_CODES = frozenset("lLnNP")


def _encode_binary_structure(layout):
    fields = []
    for entry in layout._structure_format:
        if hasattr(entry, "from_stream"):
            # A nested layout object inside a field list cannot be
            # represented in the document schema
            fields.append(_opaque(entry))
            continue
        name, format = entry[:2]
        if (
            layout._byte_order == "@"
            and not format.startswith(("<", ">", "=", "!"))
            and format[-1] in _NATIVE_SIZE_CODES
        ):
            # `@` is rewritten to `=` below; that is only size-preserving
            # for standard-size codes, so refuse to silently change the
            # width of this field
            raise ValueError(
                f"Field `{name}` uses format code `{format[-1]}`, whose size "
                f"is platform-dependent under native byte order `@`; declare "
                f"an explicit byte order (or a fixed-size code) to make this "
                f"structure serializable."
            )
        field = {"name": name, "format": format}
        hooks = _encode_field_hooks(entry[2:])
        if hooks:
            field["hooks"] = hooks
        fields.append(field)
    byte_order = layout._byte_order
    if byte_order == "@":
        # `@` is equivalent to `=` because fields are unpacked
        # individually and native-size codes are rejected above; the
        # contract forbids `@` in documents
        byte_order = "="
    return {
        "type": "BinaryStructure",
        "name": _encode_name(layout),
        "byte_order": byte_order,
        "fields": fields,
    }


def _decode_binary_structure(d):
    entries = []
    for field in d["fields"]:
        if field.get("kind") == "opaque":
            raise UnsupportedSchema(
                f"Cannot rehydrate opaque structure field "
                f"{field.get('repr')}; the document is incomplete."
            )
        entry = [field["name"], field["format"]]
        entry += _decode_field_hooks(field.get("hooks", []))
        entries.append(tuple(entry))
    return BinaryStructure(entries, byte_order=d["byte_order"], name=d["name"])


def _encode_binary_array(layout):
    res = {
        "type": "BinaryArray",
        "name": layout._name,
        "shape": encode_value(layout._shape),
        "dtype": encode_value(layout._dtype),
    }
    if layout._conversion_fun is not _identity:
        res["conversion"] = encode_value(layout._conversion_fun)
    if layout._mask_fun is not None:
        res["mask"] = encode_value(layout._mask_fun)
    return res


def _decode_binary_array(d):
    kwargs = {}
    if "conversion" in d:
        kwargs["conversion_fun"] = decode_value(d["conversion"])
    if "mask" in d:
        kwargs["mask_fun"] = decode_value(d["mask"])
    return BinaryArray(
        d["name"], decode_value(d["shape"]), decode_value(d["dtype"]), **kwargs
    )


def _encode_raw_buffer(layout):
    return {
        "type": "RawBuffer",
        "name": layout._name,
        "size": encode_value(layout._size),
        "lazy": layout._lazy,
    }


def _decode_raw_buffer(d):
    return RawBuffer(d["name"], decode_value(d["size"]), lazy=d["lazy"])


def _encode_text_buffer(layout):
    return {
        "type": "TextBuffer",
        "name": layout._name,
        "size": encode_value(layout._size),
        "encoding": layout._encoding,
    }


def _decode_text_buffer(d):
    return TextBuffer(d["name"], decode_value(d["size"]), encoding=d["encoding"])


def _encode_compound_layout(layout):
    res = {
        "type": "CompoundLayout",
        "name": _encode_name(layout),
        "structures": [layout_to_dict(s) for s in layout._structures],
    }
    if layout._context_mapper is not _identity:
        res["context_mapper"] = encode_value(layout._context_mapper)
    return res


def _decode_compound_layout(d):
    kwargs = {}
    if "context_mapper" in d:
        kwargs["context_mapper"] = decode_value(d["context_mapper"])
    return CompoundLayout(
        [layout_from_dict(s) for s in d["structures"]],
        name=decode_value(d["name"]),
        **kwargs,
    )


def _encode_if(layout):
    args = layout._args
    nb_conditions = len(args) // 2
    branches = [
        {
            "condition": encode_value(args[2 * i]),
            "layout": layout_to_dict(args[2 * i + 1]),
        }
        for i in range(nb_conditions)
    ]
    res = {"type": "If", "branches": branches}
    if len(args) > 2 * nb_conditions:
        res["default"] = layout_to_dict(args[2 * nb_conditions])
    if layout._context_mapper is not None:
        res["context_mapper"] = encode_value(layout._context_mapper)
    return res


def _decode_if(d):
    args = []
    for branch in d["branches"]:
        args.append(decode_value(branch["condition"]))
        args.append(layout_from_dict(branch["layout"]))
    if "default" in d:
        args.append(layout_from_dict(d["default"]))
    kwargs = {}
    if "context_mapper" in d:
        kwargs["context_mapper"] = decode_value(d["context_mapper"])
    return If(*args, **kwargs)


def _encode_switch(layout):
    return {
        "type": "Switch",
        "key": encode_value(layout._key),
        "cases": [
            {"value": encode_value(value), "layout": layout_to_dict(case)}
            for value, case in layout._cases.items()
        ],
        **(
            {"default": layout_to_dict(layout._default)}
            if layout._default is not None
            else {}
        ),
    }


def _decode_switch(d):
    cases = {
        decode_value(case["value"]): layout_from_dict(case["layout"])
        for case in d["cases"]
    }
    default = layout_from_dict(d["default"]) if "default" in d else None
    return Switch(decode_value(d["key"]), cases, default=default)


def _encode_skip(layout):
    res = {"type": "Skip", "size": encode_value(layout._size)}
    if layout._comment is not None:
        res["comment"] = layout._comment
    return res


def _decode_skip(d):
    return Skip(decode_value(d["size"]), comment=d.get("comment"))


def _encode_seek(layout):
    res = {"type": "Seek", "offset": encode_value(layout._offset)}
    if layout._comment is not None:
        res["comment"] = layout._comment
    return res


def _decode_seek(d):
    return Seek(decode_value(d["offset"]), comment=d.get("comment"))


def _encode_sized_chunk(layout):
    res = {
        "type": "SizedChunk",
        "name": _encode_name(layout),
        "size": encode_value(layout._size),
        "structure": layout_to_dict(layout._structure),
        "mode": layout._mode,
    }
    if layout._context_mapper is not None:
        res["context_mapper"] = encode_value(layout._context_mapper)
    return res


def _decode_sized_chunk(d):
    kwargs = {}
    if "context_mapper" in d:
        kwargs["context_mapper"] = decode_value(d["context_mapper"])
    return SizedChunk(
        decode_value(d["size"]),
        layout_from_dict(d["structure"]),
        mode=d["mode"],
        name=decode_value(d["name"]),
        **kwargs,
    )


def _encode_for(layout):
    return {
        "type": "For",
        "name": _encode_name(layout),
        "count": encode_value(layout._range),
        "structure": layout_to_dict(layout._structure),
    }


def _decode_for(d):
    return For(
        decode_value(d["count"]),
        layout_from_dict(d["structure"]),
        name=decode_value(d["name"]),
    )


def _encode_while(layout):
    body = []
    for arg in layout._args:
        if hasattr(arg, "from_stream"):
            body.append(layout_to_dict(arg))
        else:
            body.append({"condition": encode_value(arg)})
    return {"type": "While", "name": _encode_name(layout), "body": body}


def _decode_while(d):
    args = []
    for item in d["body"]:
        if "condition" in item and "type" not in item:
            args.append(decode_value(item["condition"]))
        else:
            args.append(layout_from_dict(item))
    return While(*args, name=decode_value(d["name"]))


def _encode_tlv_container(layout):
    res = {
        "type": "TLVContainer",
        "name": _encode_name(layout),
        "tag_format": layout._tag_format,
        "size_format": layout._size_format,
        "store_by_name": layout._store_by_name,
        "tags": [
            {
                "tag": tag,
                "layout": (
                    case
                    if case is None or case == "text"
                    else layout_to_dict(case)
                ),
            }
            for tag, case in layout._tag_map.items()
        ],
    }
    if layout._count is not None:
        res["count"] = encode_value(layout._count)
    if layout._container_size is not None:
        res["container_size"] = encode_value(layout._container_size)
    if layout._entry_prefix_format is not None:
        res["entry_prefix_format"] = layout._entry_prefix_format
    if layout._default is not None:
        res["default"] = layout_to_dict(layout._default)
    if layout._hex_tag_keys:
        res["hex_tag_keys"] = True
    return res


def _decode_tlv_container(d):
    tag_map = {}
    for entry in d["tags"]:
        case = entry["layout"]
        if case is not None and case != "text":
            case = layout_from_dict(case)
        tag_map[entry["tag"]] = case
    return TLVContainer(
        tag_map,
        name=decode_value(d["name"]),
        tag_format=d["tag_format"],
        size_format=d["size_format"],
        count=decode_value(d.get("count")),
        container_size=decode_value(d.get("container_size")),
        store_by_name=d["store_by_name"],
        entry_prefix_format=d.get("entry_prefix_format"),
        default=(
            layout_from_dict(d["default"]) if "default" in d else None
        ),
        hex_tag_keys=d.get("hex_tag_keys", False),
    )


def _encode_zip_container(layout):
    members = []
    for entry in layout._members:
        member_name, member_layout = entry[:2]
        optional = entry[2] if len(entry) > 2 else False
        members.append(
            {
                "member": member_name,
                "layout": layout_to_dict(member_layout),
                "optional": optional,
            }
        )
    res = {
        "type": "ZipContainer",
        "name": _encode_name(layout),
        "members": members,
    }
    if layout._stream_filter is not None:
        res["stream_filter"] = encode_value(layout._stream_filter)
    return res


def _decode_zip_container(d):
    members = [
        (m["member"], layout_from_dict(m["layout"]), m["optional"])
        for m in d["members"]
    ]
    return ZipContainer(
        members,
        name=decode_value(d["name"]),
        stream_filter=decode_value(d.get("stream_filter")),
    )


def _encode_xml_structure(layout):
    return {
        "type": "XMLStructure",
        "name": _encode_name(layout),
        "converters": {
            tag: encode_value(converter)
            for tag, converter in layout._converters.items()
        },
    }


def _decode_xml_structure(d):
    return XMLStructure(
        name=decode_value(d["name"]),
        converters={
            tag: decode_value(converter)
            for tag, converter in d["converters"].items()
        },
    )


def _encode_check(layout):
    res = {
        "type": "Check",
        "condition": encode_value(layout._condition),
        "error": _encode_error(layout._exception),
    }
    if layout._comment is not None:
        res["comment"] = layout._comment
    return res


def _decode_check(d):
    return Check(
        decode_value(d["condition"]),
        exception=_decode_error(d["error"]),
        comment=d.get("comment"),
    )


def _encode_text_line(layout):
    return {
        "type": "TextLine",
        "name": layout._name,
        "encoding": layout._encoding,
    }


def _decode_text_line(d):
    return TextLine(d["name"], encoding=d["encoding"])


def _encode_text_header(layout):
    res = {
        "type": "TextHeader",
        "name": _encode_name(layout),
        "encoding": layout._encoding,
        "separator": layout._separator,
        "strict": layout._strict,
    }
    if layout._key_width is not None:
        res["key_width"] = layout._key_width
    if layout._comment_prefixes:
        res["comment_prefixes"] = list(layout._comment_prefixes)
    if layout._sections is not None:
        res["sections"] = layout._sections
    if layout._section_key is not None:
        res["section_key"] = layout._section_key
        res["sections_name"] = layout._sections_name
    if layout._terminator is not None:
        res["terminator"] = layout._terminator
    if layout._stop_key is not None:
        res["stop_key"] = layout._stop_key
    if layout._size is not None:
        res["size"] = encode_value(layout._size)
    if layout._converters:
        res["converters"] = {
            key: encode_value(converter)
            for key, converter in layout._converters.items()
        }
    return res


def _decode_text_header(d):
    return TextHeader(
        name=decode_value(d["name"]),
        encoding=d["encoding"],
        separator=d["separator"],
        key_width=d.get("key_width"),
        comment_prefixes=tuple(d.get("comment_prefixes", ())),
        sections=d.get("sections"),
        section_key=d.get("section_key"),
        sections_name=d.get("sections_name", "sections"),
        terminator=d.get("terminator"),
        stop_key=d.get("stop_key"),
        size=decode_value(d.get("size")),
        converters={
            key: decode_value(converter)
            for key, converter in d.get("converters", {}).items()
        },
        strict=d["strict"],
    )


def _encode_text_matrix(layout):
    res = {
        "type": "TextMatrix",
        "name": layout._name,
        "shape": encode_value(layout._shape),
        "encoding": layout._encoding,
        "bad_markers": list(layout._bad_markers),
    }
    if layout._conversion_fun is not _identity:
        res["conversion"] = encode_value(layout._conversion_fun)
    return res


def _decode_text_matrix(d):
    kwargs = {}
    if "conversion" in d:
        kwargs["conversion_fun"] = decode_value(d["conversion"])
    return TextMatrix(
        d["name"],
        decode_value(d["shape"]),
        encoding=d["encoding"],
        bad_markers=tuple(d["bad_markers"]),
        **kwargs,
    )


def _encode_zlib_block_chain(layout):
    return {
        "type": "ZlibBlockChain",
        "name": layout._name,
        "prefix_format": layout._prefix_format,
        "min_decompressed_size": layout._min_decompressed_size,
    }


def _decode_zlib_block_chain(d):
    return ZlibBlockChain(
        d["name"],
        prefix_format=d["prefix_format"],
        min_decompressed_size=d["min_decompressed_size"],
    )


_ENCODERS = {
    BinaryStructure: _encode_binary_structure,
    BinaryArray: _encode_binary_array,
    RawBuffer: _encode_raw_buffer,
    TextBuffer: _encode_text_buffer,
    CompoundLayout: _encode_compound_layout,
    Check: _encode_check,
    If: _encode_if,
    Switch: _encode_switch,
    Skip: _encode_skip,
    Seek: _encode_seek,
    SizedChunk: _encode_sized_chunk,
    For: _encode_for,
    While: _encode_while,
    TextLine: _encode_text_line,
    TextHeader: _encode_text_header,
    TextMatrix: _encode_text_matrix,
    TLVContainer: _encode_tlv_container,
    ZipContainer: _encode_zip_container,
    XMLStructure: _encode_xml_structure,
    ZlibBlockChain: _encode_zlib_block_chain,
}

_DECODERS = {
    "BinaryStructure": _decode_binary_structure,
    "BinaryArray": _decode_binary_array,
    "RawBuffer": _decode_raw_buffer,
    "TextBuffer": _decode_text_buffer,
    "CompoundLayout": _decode_compound_layout,
    "Check": _decode_check,
    "If": _decode_if,
    "Switch": _decode_switch,
    "Skip": _decode_skip,
    "Seek": _decode_seek,
    "SizedChunk": _decode_sized_chunk,
    "For": _decode_for,
    "While": _decode_while,
    "TextLine": _decode_text_line,
    "TextHeader": _decode_text_header,
    "TextMatrix": _decode_text_matrix,
    "TLVContainer": _decode_tlv_container,
    "ZipContainer": _decode_zip_container,
    "XMLStructure": _decode_xml_structure,
    "ZlibBlockChain": _decode_zlib_block_chain,
}

# Layout node types that require a capability beyond `core`, per the
# format description contract
_NODE_CAPABILITIES = {
    "ZipContainer": "zip",
    "XMLStructure": "xml",
    "ZlibBlockChain": "zlib",
    "TextLine": "text",
    "TextHeader": "text",
    "TextMatrix": "text",
}


def layout_to_dict(layout):
    """
    Serialize a layout tree to the contract's JSON representation.

    Hooks that are plain Python callables serialize to opaque markers;
    use `count_opaque` to check whether the result is complete.
    """
    try:
        encoder = _ENCODERS[type(layout)]
    except KeyError:
        # E.g. a bare callable used as a structure factory
        return _opaque(layout)
    return encoder(layout)


def layout_from_dict(d):
    """Rehydrate a layout tree from its JSON representation."""
    if not isinstance(d, dict) or "type" not in d:
        raise UnsupportedSchema(f"Cannot decode layout node: {d!r}")
    try:
        decoder = _DECODERS[d["type"]]
    except KeyError:
        raise UnsupportedSchema(f"Unknown layout node type `{d['type']}`.")
    return decoder(d)


def count_opaque(document):
    """Count opaque (non-serializable) hooks in an encoded document."""
    if isinstance(document, dict):
        if document.get("kind") == "opaque":
            return 1
        return sum(count_opaque(v) for v in document.values())
    elif isinstance(document, list):
        return sum(count_opaque(v) for v in document)
    else:
        return 0


def capabilities(document):
    """
    Compute the sorted capability list of an encoded document, from its
    layout node types and registry function calls.
    """
    found = {"core"}

    def walk(obj):
        if isinstance(obj, dict):
            node_type = obj.get("type")
            if node_type in _NODE_CAPABILITIES:
                found.add(_NODE_CAPABILITIES[node_type])
            if obj.get("kind") == "call":
                capability = expr.FUNCTION_CAPABILITIES.get(obj.get("name"))
                if capability is not None:
                    found.add(capability)
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(document)
    return sorted(found)


###
# Channel bindings and full format description documents
###


def channel_bindings_to_dict(bindings):
    """Serialize the `_channel_bindings` of a declarative reader."""
    return [encode_value(binding) for binding in bindings]


def channel_bindings_from_dict(document):
    """Rehydrate channel bindings from their JSON representation."""
    return [decode_value(binding) for binding in document]


def reader_description(reader_class):
    """
    Build the complete format description document for a declarative
    reader class, per the format description contract.
    """
    import base64

    layout = layout_to_dict(reader_class._file_layout)
    channels = channel_bindings_to_dict(reader_class._channel_bindings or [])
    magic = [
        {
            "offset": offset,
            "bytes": base64.b64encode(magic_bytes).decode("ascii"),
        }
        for offset, magic_bytes in (getattr(reader_class, "_magic", None) or [])
    ]
    document = {
        "schema_version": SCHEMA_VERSION,
        "format": {
            "id": reader_class._format,
            "name": reader_class._name,
            "description": (reader_class._description or "").strip(),
            "file_extensions": reader_class._file_extensions,
            "mime_types": reader_class._mime_types,
            "magic": magic,
        },
        "layout": layout,
        "channels": channels,
    }
    document["capabilities"] = capabilities(document)
    return document


def reader_from_dict(document):
    """
    Create a reader class from a format description document. This is the
    generic, data-driven counterpart of the hand-authored declarative
    reader classes; it is also how the round-trip CI gate proves that a
    document contains all format knowledge.
    """
    import base64

    schema_version = document.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise UnsupportedSchema(
            f"This engine implements schema version {SCHEMA_VERSION}, but the "
            f"document declares version {schema_version}."
        )
    layout = layout_from_dict(document["layout"])
    bindings = channel_bindings_from_dict(document.get("channels", []))
    format_info = document.get("format", {})
    magic = [
        (entry["offset"], base64.b64decode(entry["bytes"]))
        for entry in format_info.get("magic", [])
    ]

    class DescribedReader(DeclarativeReaderBase):
        _file_layout = layout
        _channel_bindings = bindings
        _format = format_info.get("id")
        _name = format_info.get("name")
        _description = format_info.get("description", "")
        _file_extensions = format_info.get("file_extensions")
        _mime_types = format_info.get("mime_types")
        _magic = magic or None

    return DescribedReader
