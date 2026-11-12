import pytest
from opensees.openseespy import Model
from opensees.tcl import InterpreterError


TAIL = "-Xo 0.3 -D 0.67 -W 0.064 -R 2.5 -gamma 0.17 -theta 0.02 -beta 0.67 -alpha 0.27 -T -0.3 -tol 1e-8"
POSITIONAL = "3 0 40 66.667 0.3 0.67 0.064 2.5 0.17 0.02 0.67 0.27 -0.3 1e-8"


def stress(model, tag, ndm=3, scale=1.0):
    values = [-0.0001, -0.0002, 0, 0.00001, 0, 0] if ndm == 3 else [-0.0001, -0.0002, 0]
    strain = " ".join(str(scale*value) for value in values)
    return tuple(map(float, model.eval(
        f"invoke Material {tag} {{material::update {strain}; material::stress}}"
    ).split()))


@pytest.mark.parametrize("kind", ["CapPlasticity"]) #TODO, "DruckerPlasticity02"])
@pytest.mark.parametrize("command", ["nDMaterial"])
@pytest.mark.parametrize("scale", [1.0, 10.0])
def test_elastic_keywords_and_positional_equivalence(kind, command, scale):
    model = Model(ndm=3, ndf=3)
    model.eval(f"{command} {kind} 1 {POSITIONAL}")
    expected = stress(model, 1, scale=scale)
    # In particular, -lambda must remain Lame's first elastic parameter,
    # independently of -gamma in TAIL.
    pairs = ["-G 40 -K 66.667", "-G 40 -lambda 40.0003333333333",
             "-E 100.00008333298612 -nu 0.2500010416623264"]
    for tag, pair in enumerate(pairs, 2):
        model.eval(f"{command} {kind} {tag} {pair} -density 0 {TAIL}")
        assert stress(model, tag, scale=scale) == pytest.approx(expected, rel=1e-10, abs=1e-12)

    model.eval(f"{command} {kind} 5 3 0 -G 40 -K 66.667 "
               "0.3 0.67 0.064 2.5 0.17 0.02 0.67 0.27 -0.3 1e-8")
    assert stress(model, 5, scale=scale) == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["CapPlasticity"])#TODO, "DruckerPlasticity02"])
def test_dimensions_and_legacy_defaults(kind):
    model = Model(ndm=2, ndf=2)
    model.eval(f"nDMaterial {kind} 1 2 0 40 66.667")
    dimension = 2 if kind == "CapPlasticity" else 3
    assert len(stress(model, 1, dimension)) == (3 if dimension == 2 else 6)
    model.eval(f"nDMaterial {kind} 2 -ndm 2 -G 40 -K 66.667 {TAIL}")
    assert len(stress(model, 2, dimension)) == (3 if dimension == 2 else 6)


@pytest.mark.parametrize("kind", ["CapPlasticity"])#TODO, "DruckerPlasticity02"])
@pytest.mark.parametrize("args", ["", "3 0 40", "-G 40", "-G 40 -K",
    "-G 40 -K invalid", "-G 40 -G 50", "-G 40 -K 66 -E 100",
    "-G 40 -K 66 -gamma", "-G 40 -K 66 -gamma invalid",
    "-G 40 -K 66 -unknown 1", "-G 40 -K 66 -ndm 4",
    POSITIONAL + " 999"])
def test_invalid_arguments(kind, args):
    model = Model(ndm=3, ndf=3)
    with pytest.raises(InterpreterError):
        model.eval(f"nDMaterial {kind} 1 {args}")
    # A failed parse must not insert a material with the supplied tag.
    model.eval(f"nDMaterial {kind} 1 {POSITIONAL}")


def test_duplicate_tag_rejected():
    model = Model(ndm=3, ndf=3)
    model.eval(f"nDMaterial CapPlasticity 1 {POSITIONAL}")
    with pytest.raises(InterpreterError):
        model.eval(f"nDMaterial CapPlasticity 1 {POSITIONAL}")
    assert len(stress(model, 1)) == 6

