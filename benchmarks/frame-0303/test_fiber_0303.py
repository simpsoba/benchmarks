#
#
#
import xara
import jax
jax.config.update("jax_enable_x64", True)
from xara.para import ChainModel
from xara.auto import StaticAnalysis, Array
from xsection.library._rectangle import create_fibers

import pytest


nx, ny = 3,3
L = 10.0
H = 300  # Load applied at the top node
nu = 0.3

def test_bwd():

    # Model factory
    def create_model(L, E, fibers: Array[:, ("y", "z", "area")]):

        m = ChainModel(ndm=3, ndf=6, parameters=2+len(fibers)*3)

        m.node(1,  (0,0,0))
        m.node(2,  (L,0,0))
        m.fix( 1,  (1,1,1,1,1,1))

        m.material("ElasticIsotropic", 1, E, nu)

        m.section("ShearFiber", 1, mixed_type="None")
        for fiber in fibers:
            m.fiber(*fiber, material=1, section=1)


        m.geomTransf("Linear", 1,(0,0,1))

        m.element("ForceFrame", 1, (1,2),
                section=1,
                transform=1, shear=1)


        m.pattern("Plain", 1, "Linear")
        m.load(2, (0.0,0,H, 0.0,0.0,0.0), pattern=1)

        m.constraints("Plain")
        m.system("ProfileSPD")
        m.integrator("LoadControl", 1)
        m.analysis("Static")
        return m


    # Physical Parameters
    E  = 200e6           # Young's modulus
    G  = E / (2 * (1 + nu))
    d  = 0.35           # Web height
    b  = 0.35           # Flange width
    
    Iy = d**3 * b / 12.0
    Iz = b**3 * d / 12.0
    A = d * b

    fibers = create_fibers(b, d, nx,ny)

    print("A = " + str(A))
    print("As = " + str(sum(fiber[2] for fiber in fibers)))

    print("Iy = " + str(Iy))
    print("Iyf = " + str(sum(fiber[2]*fiber[1]**2 for fiber in fibers)))
    print("Iz = " + str(Iz))
    print("Izf = " + str(sum(fiber[2]*fiber[0]**2 for fiber in fibers)))

    # Bind function for node=2,dof=3
    analyze_1 = StaticAnalysis(create_model, output=(2, 3))

    uy   = analyze_1(L,E, fibers)  
    print("uy =", uy)


    #
    # Chain Rule
    #
    def analyze_2(L, E, d, b):
        fibers = create_fibers(b, d, nx, ny)
        return analyze_1(L, E, fibers)

    du = jax.jacrev(analyze_2, (0,1,2,3))
    dL, dE, dh, db = map(float,du(L, E, d, b))
    print("∂u/∂E  =",  dE)
    print("∂u/∂h =",  dh)
    print("∂u/∂b  =", db)

    dI = -H*L**3/(3*E*Iy**2)
    dA = -H*L/(G*A**2)
    dIdh, dIdb = jax.grad(lambda d, b: b*d**3/12.0, (0,1))(d,b)
    dAdh, dAdb = jax.grad(lambda d, b: b*d, (0,1))(d,b)

    # dudI = -H*L**3/(3*E*I**2)
    # dudA = -H*L/(G*A**2)
    assert uy == pytest.approx(H*L**3/(3*E*Iy) + H*L/(G*A))
    assert dh == pytest.approx(dIdh*dI + dAdh*dA, rel=1e-3)
    assert db == pytest.approx(dIdb*dI + dAdb*dA, rel=1e-3)
    assert dL == pytest.approx(H*L**2/(E*Iy) + H/(G*A), rel=1e-3)
    assert dE == pytest.approx(-H*L**3/(3*E**2*Iy)- (H*L/(G**2*A))*G/E, rel=1e-3)# 



def test_fwd():

    # 1. Define model builder
    def create_model(L, E, fibers: Array[:, ("y", "z", "area")]):

        m = ChainModel(ndm=3, ndf=6, parameters=2+len(fibers)*3)

        m.node(1,  (0,0,0))
        m.node(2,  (L,0,0))
        m.fix( 1,  (1,1,1,1,1,1))

        m.material("ElasticIsotropic", 1, E, nu)

        m.section("ShearFiber", 1, mixed_type="None")
        for fiber in fibers:
            m.fiber(*fiber, material=1, section=1)


        m.geomTransf("Linear", 1,(0,0,1))

        m.element("ForceFrame", 1, (1,2),
                section=1,
                transform=1, shear=1)


        m.pattern("Plain", 1, "Linear")
        m.load(2, (0.0,0,H, 0.0,0.0,0.0), pattern=1)

        m.constraints("Plain")
        m.system("ProfileSPD")
        m.integrator("LoadControl", 1)
        m.analysis("Static")
        return m


    # Physical Parameters
    E  = 200e6           # Young's modulus
    G  = E / (2 * (1 + nu))
    d  = 0.35           # Web height
    b  = 0.35           # Flange width
    
    Iy = d**3 * b / 12.0
    Iz = b**3 * d / 12.0
    A = d * b

    fibers = create_fibers(b, d, nx,ny)


    # Bind function for node=2,dof=3
    analyze_1 = StaticAnalysis(create_model, output=(2, 3))

    uy   = analyze_1(L,E, fibers)  
    print("uy =", uy)


    #
    # Chain Rule
    #
    def analyze_2(L, E, d, b):
        fibers = create_fibers(b, d, nx, ny)
        return analyze_1(L, E, fibers)

    du = jax.jacfwd(analyze_2, (0,1,2,3))
    dL, dE, dh, db = map(float,du(L, E, d, b))


    #
    # Verify
    #
    print("∂u/∂E  =",  dE)
    print("∂u/∂h =",  dh)
    print("∂u/∂b  =", db)

    print("A = " + str(A))
    print("As = " + str(sum(fiber[2] for fiber in fibers)))

    print("Iy = " + str(Iy))
    print("Iyf = " + str(sum(fiber[2]*fiber[1]**2 for fiber in fibers)))
    print("Iz = " + str(Iz))
    print("Izf = " + str(sum(fiber[2]*fiber[0]**2 for fiber in fibers)))

    dI = -H*L**3/(3*E*Iy**2)
    dA = -H*L/(G*A**2)
    dIdh, dIdb = jax.grad(lambda d, b: b*d**3/12.0, (0,1))(d,b)
    dAdh, dAdb = jax.grad(lambda d, b: b*d, (0,1))(d,b)

    assert uy == pytest.approx(H*L**3/(3*E*Iy) + H*L/(G*A))
    assert dh == pytest.approx(dIdh*dI + dAdh*dA, rel=1e-4)
    assert db == pytest.approx(dIdb*dI + dAdb*dA, rel=1e-4)
    assert dL == pytest.approx(H*L**2/(E*Iy) + H/(G*A), rel=1e-3)
    assert dE == pytest.approx(-H*L**3/(3*E**2*Iy)- (H*L/(G**2*A))*G/E, rel=1e-3)# 


def test_grad():

    # 1. Define model builder
    def create_model(L, E, fibers: Array[:, ("y", "z", "area")]):

        m = ChainModel(ndm=3, ndf=6, parameters=2+len(fibers)*3)

        m.node(1,  (0,0,0))
        m.node(2,  (L,0,0))
        m.fix( 1,  (1,1,1,1,1,1))

        m.material("ElasticIsotropic", 1, E, nu)

        # Define section
        m.section("ShearFiber", 1, mixed_type="None")
        # Populate section with fibers
        for fiber in fibers:
            m.fiber(*fiber, material=1, section=1)


        m.geomTransf("Linear", 1,(0,0,1))

        m.element("ForceFrame", 1, (1,2),
                section=1,
                transform=1,
                shear=1)


        m.pattern("Plain", 1, "Linear")
        m.load(2, (0.0,0,H, 0.0,0.0,0.0), pattern=1)

        m.constraints("Plain")
        m.system("ProfileSPD")
        m.integrator("LoadControl", 1)
        m.analysis("Static")
        return m


    # Physical Parameters
    E  = 200e6           # Young's modulus
    G  = E / (2 * (1 + nu))
    d  = 0.35           # Web height
    b  = 0.35           # Flange width
    fibers = create_fibers(b, d, nx,ny)

    Iy = d**3 * b / 12.0
    Iz = b**3 * d / 12.0
    A  = d * b

    # print("A = " + str(A))
    # print("As = " + str(sum(fiber[2] for fiber in fibers)))

    # print("Iy = " + str(Iy))
    # print("Iyf = " + str(sum(fiber[2]*fiber[1]**2 for fiber in fibers)))
    # print("Iz = " + str(Iz))
    # print("Izf = " + str(sum(fiber[2]*fiber[0]**2 for fiber in fibers)))

    # Bind function for node=2,dof=3
    analyze_1 = StaticAnalysis(create_model, output=(2, 3))

    uy   = analyze_1(L,E, fibers)

    #
    # Chain Rule
    #
    def analyze_2(L, E, d, b):
        fibers = create_fibers(b, d, nx, ny)
        return analyze_1(L, E, fibers)

    du = jax.grad(analyze_2, (0,1,2,3))
    dL, dE, dh, db = map(float,du(L, E, d, b))

    dI = -H*L**3/(3*E*Iy**2)
    dA = -H*L/(G*A**2)
    dIdh, dIdb = jax.grad(lambda d, b: b*d**3/12.0, (0,1))(d,b)
    dAdh, dAdb = jax.grad(lambda d, b: b*d, (0,1))(d,b)

    assert uy == pytest.approx(H*L**3/(3*E*Iy) + H*L/(G*A))
    assert dh == pytest.approx(dIdh*dI + dAdh*dA, rel=1e-5)
    assert db == pytest.approx(dIdb*dI + dAdb*dA, rel=1e-5)
    assert dL == pytest.approx(H*L**2/(E*Iy) + H/(G*A), rel=1e-3)
    assert dE == pytest.approx(-H*L**3/(3*E**2*Iy)- (H*L/(G**2*A))*G/E, rel=1e-5)# 


    def _print(name, val, ref):
        if isinstance(name, tuple):
            name = f"∂{name[0]}/∂{name[1]}"

        print(name, "\t", val, "\t", ((val-ref)/ref)*100, "%")

    _print("u", uy, H*L**3/(3*E*Iy) + H*L/(G*A))
    _print(("u", "E"), dE, -H*L**3/(3*E**2*Iy)- (H*L/(G**2*A))*G/E)
    _print(("u", "h"), dh, dIdh*dI + dAdh*dA)
    _print(("u", "b"), db, dIdb*dI + dAdb*dA)
    _print(("u", "L"), dL, H*L**2/(E*Iy) + H/(G*A))

    # print("∂u/∂E =", dE, -H*L**3/(3*E**2*Iy)- (H*L/(G**2*A))*G/E)
    # print("∂u/∂h =", dh, dIdh*dI + dAdh*dA)
    # print("∂u/∂b =", db, dIdb*dI + dAdb*dA)
    # print("∂u/∂L =", dL, H*L**2/(E*Iy) + H/(G*A))


if __name__ == "__main__":
    test_grad()
