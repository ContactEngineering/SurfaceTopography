from SurfaceTopography.IO.binary import BinaryStructure, Convert, Validate
from SurfaceTopography.IO.Reader import CompoundLayout, For, If, SizedChunk, Skip


def test_binary_structure_to_dict():
    bs = BinaryStructure(
        [
            ("magic", "4s", Validate(b"1234", ValueError)),
            ("version", "I", None, Convert(int)),
            ("data", "f"),
        ],
        byte_order="<",
        name="header",
    )

    d = bs.to_dict()
    assert d["type"] == "BinaryStructure"
    assert d["name"] == "header"
    assert d["byte_order"] == "<"
    assert len(d["structure"]) == 3
    assert d["structure"][0]["name"] == "magic"
    assert d["structure"][0]["format"] == "4s"
    assert d["structure"][0]["validate"] == {"type": "Validate", "value": b"1234"}
    assert d["structure"][1]["name"] == "version"
    assert d["structure"][1]["format"] == "I"
    assert d["structure"][1]["convert"] == {
        "type": "Convert",
        "fun": "callable_convert",
    }
    assert d["structure"][2]["name"] == "data"
    assert d["structure"][2]["format"] == "f"


def test_binary_structure_with_lambdas_to_dict():
    bs = BinaryStructure(
        [
            ("magic", "4s", lambda x, context: x == b"1234"),
            ("version", "I", None, lambda x: int(x)),
        ],
        name=lambda context: "dynamic_name",
    )

    d = bs.to_dict()
    assert d["name"] == {"__type__": "lambda", "source": "callable_name"}
    assert d["structure"][0]["validate"] == {
        "type": "lambda",
        "source": "callable_validator",
    }
    assert d["structure"][1]["convert"] == {
        "type": "lambda",
        "source": "callable_converter",
    }


def test_compound_layout_to_dict():
    s1 = BinaryStructure([("a", "I")], name="s1")
    s2 = BinaryStructure([("b", "f")], name="s2")
    cl = CompoundLayout([s1, s2], name="compound")

    d = cl.to_dict()
    assert d["type"] == "CompoundLayout"
    assert d["name"] == "compound"
    assert len(d["structures"]) == 2
    assert d["structures"][0]["name"] == "s1"
    assert d["structures"][1]["name"] == "s2"


def test_if_to_dict():
    s1 = BinaryStructure([("a", "I")], name="s1")
    s2 = BinaryStructure([("b", "f")], name="s2")
    # If takes args like (condition1, structure1, condition2, structure2, ..., default_structure)
    obj = If(lambda ctx: True, s1, s2)

    d = obj.to_dict()
    assert d["type"] == "If"
    assert len(d["args"]) == 3
    assert d["args"][0] == {"type": "lambda", "source": "callable_arg"}
    assert d["args"][1]["name"] == "s1"
    assert d["args"][2]["name"] == "s2"


def test_skip_to_dict():
    obj = Skip(size=10, comment="skip 10 bytes")
    d = obj.to_dict()
    assert d["type"] == "Skip"
    assert d["size"] == 10
    assert d["comment"] == "skip 10 bytes"

    obj_lambda = Skip(size=lambda ctx: 20)
    d_lambda = obj_lambda.to_dict()
    assert d_lambda["size"] == {"__type__": "lambda", "source": "callable_size"}


def test_sized_chunk_to_dict():
    s1 = BinaryStructure([("a", "I")], name="s1")
    obj = SizedChunk(size=100, structure=s1, name="chunk")
    d = obj.to_dict()
    assert d["type"] == "SizedChunk"
    assert d["name"] == "chunk"
    assert d["size"] == 100
    assert d["structure"]["name"] == "s1"

    obj_lambda = SizedChunk(size=lambda ctx: 200, structure=s1)
    d_lambda = obj_lambda.to_dict()
    assert d_lambda["size"] == {"__type__": "lambda", "source": "callable_size"}


def test_for_to_dict():
    s1 = BinaryStructure([("a", "I")], name="s1")
    obj = For(range=10, structure=s1, name="loop")
    d = obj.to_dict()
    assert d["type"] == "For"
    assert d["name"] == "loop"
    assert d["range"] == 10
    assert d["structure"]["name"] == "s1"

    obj_lambda = For(range=lambda ctx: range(5), structure=s1, name="loop_lambda")
    d_lambda = obj_lambda.to_dict()
    assert d_lambda["range"] == {"__type__": "lambda", "source": "callable_range"}


def test_layout_with_name_base_to_dict():
    from SurfaceTopography.IO.binary import LayoutWithNameBase

    obj = LayoutWithNameBase()
    obj._name = "base"
    d = obj.to_dict()
    assert d["type"] == "LayoutWithNameBase"
    assert d["name"] == "base"

    obj_lambda = LayoutWithNameBase()
    obj_lambda._name = lambda ctx: "dynamic"
    d_lambda = obj_lambda.to_dict()
    assert d_lambda["name"] == {"__type__": "lambda", "source": "callable_name"}
