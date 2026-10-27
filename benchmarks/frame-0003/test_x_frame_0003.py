#
# Column with distributed loads, linear geometry
#

import pytest
from xara_prism import AISC_360_C1, check_span


rtol = {"u_tran": 1e-6, "u_long": 1e-12}

analysis_options={
    "test": ("Energy", 1e-18, 2, 0)
}



@pytest.mark.parametrize("element", ["ForceFrame", "PrismFrame", "EulerFrame"])
def test_aisc_euler(element):

    column = AISC_360_C1(shear=0)


    for basis in ["local", "global"]:
        loading = dict(basis=basis) if basis is not None else None

        check_span(
            column.analyze(3, element, "Linear", ne=4, steps=2, loading_options=loading),
            column.solution("linear"),
            time=1,
            rtol=rtol
        )

@pytest.mark.parametrize("element", ["ForceFrame", "PrismFrame", "ShearFrame"])
def test_aisc_shear(element):

    column = AISC_360_C1(shear=1)

    # Lagrange-interpolated element; needs at least nen=5 (quartic displacement)
    # for uniform load solution
    nen = 5 if element == "ShearFrame" else 2


    for basis in ["local", "global"]:
        loading = dict(basis=basis) if basis is not None else None

        check_span(
            column.analyze(3, element, "Linear", ne=4, nen=nen, steps=2, loading_options=loading),
            column.solution("linear"),
            time=1,
            rtol=rtol
        )

if __name__ == "__main__":
    for element in ["ForceFrame", "PrismFrame", "EulerFrame"]:
        test_aisc_euler(element)
    for element in ["ForceFrame", "PrismFrame", "ShearFrame"]:
        test_aisc_shear(element)