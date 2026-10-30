#
# Vertical cantilever with terminal loads implemented with FrameLoad
#
import xara
from xara.load import FrameLoad, NodalLoad
import pytest
import numpy as np

L = 140
ne = 2


E = 29e3
G = 11e3
d = 12
b = 6
Iy = b*d**3/12
Iz = d*b**3/12
J  = Iy + Iz
A  = 1e3
Ay  = 5/6*b*d
Az  = Ay

#     V.    V.      N
F = [1000, 2000,   60]

M = [10.,30.,15.0]
e = (b/3, d/3)

# Global basis vectors
e_major, e_minor, e_axial = np.eye(3)
# Local basis vectors
i_major = np.array([0., 1., 0.])
i_minor = np.array([0., 0., 1.])
i_axial = np.array([1., 0., 0.])


def create_prism(shape, shear=0, load_type="global", element="ForceFrame"):

    model = xara.Model(ndm=3, ndf=6)

    model.node(1, (0, 0, 0))
    model.node(2, (0, 0, L))

    model.fix(1, (1, 1, 1, 1, 1, 1))

    transform = "Linear"

    model.geomTransf(transform, 1, (0, 1, 0))

    section = xara.FrameSection("Elastic", **shape)
    model.section(section)

    model.element(element, 1, (1, 2), 
                  section=section, 
                  shear=shear,
                  transform=1)

    if load_type == "node":
        f = F
        r = e[0]*e_major + e[1]*e_minor
        m = M + np.cross(r, F)
        model.pattern(xara.StaticPattern(
            NodalLoad(model, {
                2: (*f, *m)
            })                   
        ))

    else:
        if load_type == "global":
            f = F
            m = M
        elif load_type == "local":
            f = ( np.dot(F, e_axial)*i_axial
                + np.dot(F, e_minor)*i_minor
                + np.dot(F, e_major)*i_major).tolist()
            m = ( np.dot(M, e_axial)*i_axial
                + np.dot(M, e_minor)*i_minor
                + np.dot(M, e_major)*i_major).tolist()

        model.pattern(xara.StaticPattern(
            FrameLoad(model,
                    shape="Point",
                    elements=[1],
                    offset=(1, *e),
                    basis = load_type,
                    force=f,
                    couple=m,
            )                   
        ))

    analysis = xara.StaticAnalysis(model)
    analysis.analyze()
    return model


def solution(shape, shear=False):

    FX, FY, N = F
    r = e[0]*e_major + e[1]*e_minor
    MY, MZ, T = M + np.cross(r, F)

    EIy = shape["E"]*shape["Iy"]
    EIz = shape["E"]*shape["Iz"]
    EA  = shape["E"]*shape["A"]
    GJ  = shape["G"]*shape["J"]

    UX = FX*L**3/(3*EIz) + MZ*L**2/(2*EIz)
    UY = FY*L**3/(3*EIy) - MY*L**2/(2*EIy)
    uz = N*L/EA

    rx = MY*L/(EIy) - FY*L**2/(2*EIy)
    ry = MZ*L/(EIz) + FX*L**2/(2*EIz)
    rz = T*L/GJ

    if shear:
        GAy = shape["G"]*shape["Ay"]
        GAz = shape["G"]*shape["Az"]

        UX += FX*L/GAy
        UY += FY*L/GAz

    return list(map(float, [UX, UY, uz, rx, ry, rz]))



@pytest.mark.parametrize("element", ["ForceFrame", "PrismFrame", "EulerFrame"]) # TODO("EulerFrame")
@pytest.mark.parametrize("load_type", ["global", "local", "node"])
def test_terminal_loads_euler(load_type, element):
    shape = {
        "E":  E,
        "G":  G,
        "A":  A,
        "Iy": Iy,
        "Iz": Iz,
        "J":  J
    }
    model = create_prism(shape, 
                         load_type=load_type, 
                         shear=0,
                         element=element)
    Umodel = model.nodeDisp(2)
    Usoln = solution(shape)

    print("    Solution: ", Usoln)
    print("    Result:   ", Umodel)
    assert Umodel[0] == pytest.approx(Usoln[0], abs=1e-10)
    assert Umodel[1] == pytest.approx(Usoln[1], abs=1e-10)
    assert Umodel[2] == pytest.approx(Usoln[2], abs=1e-10)

    assert Umodel[3] == pytest.approx(Usoln[3], abs=1e-10)
    assert Umodel[4] == pytest.approx(Usoln[4], abs=1e-10)
    assert Umodel[5] == pytest.approx(Usoln[5], abs=1e-10)




@pytest.mark.parametrize("element", ["ForceFrame", "LagrangeFrame", "PrismFrame"])
@pytest.mark.parametrize("load_type", ["global", "local", "node"])
def test_terminal_loads_shear(load_type, element):
    shape = {
        "E":  E,
        "G":  G,
        "A":  A,
        "Ay": Ay,
        "Az": Az,
        "Iy": Iy,
        "Iz": Iz,
        "J":  J
    }
    model = create_prism(shape, 
                         load_type=load_type, 
                         shear=1, 
                         element=element)
    Umodel = model.nodeDisp(2)
    Usoln = solution(shape, shear=True)

    print("   Solution: ", Usoln)
    print("   Result:   ", Umodel)
    for i in range(6):
        print(f"      U[{i}]: ", (Umodel[i] - Usoln[i])/Usoln[i]*100, "%")

    # Reduce tolerance since 2-node shear element will be poor
    if "Lagrange" in element:
        utol = 0.3
        rtol = 1e-8 if load_type == "node" else utol
        assert Umodel[0] == pytest.approx(Usoln[0], rel=utol)
        assert Umodel[1] == pytest.approx(Usoln[1], rel=utol)
        # Axial extension is expected to be accurate even with shear elements
        assert Umodel[2] == pytest.approx(Usoln[2], rel=1e-6)

        assert Umodel[3] == pytest.approx(Usoln[3], rel=rtol)
        assert Umodel[4] == pytest.approx(Usoln[4], rel=rtol)
        # TODO(LagrangeFrame): Torsional rotation
        if load_type == "node":
            assert Umodel[5] == pytest.approx(Usoln[5], rel=1e-6)
    else:
        tol = 1e-10
        assert Umodel[0] == pytest.approx(Usoln[0], abs=tol)
        assert Umodel[1] == pytest.approx(Usoln[1], abs=tol)
        assert Umodel[2] == pytest.approx(Usoln[2], abs=tol)

        assert Umodel[3] == pytest.approx(Usoln[3], abs=tol)
        assert Umodel[4] == pytest.approx(Usoln[4], abs=tol)
        # Torsional rotation
        assert Umodel[5] == pytest.approx(Usoln[5], abs=tol)


if __name__ == "__main__":
    for load_type in ["node", "global", "local"]:

        print(f"\nRunning load_type={load_type}")
        for element in ["ForceFrame", "PrismFrame", "EulerFrame"]:
            print(f"  Running element={element}")
            test_terminal_loads_euler(load_type, element)

        print()
        for element in "ForceFrame", "PrismFrame", "LagrangeFrame":
            print(f"  Running element={element}")
            test_terminal_loads_shear(load_type, element)
            print()
        print()

