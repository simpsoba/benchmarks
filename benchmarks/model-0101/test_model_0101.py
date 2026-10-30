

import xara
import scipy.linalg
import numpy as np

def CreateH8(element):
    def f():
        print("Element: ", element)
        model = xara.Model(ndm=3, ndf=3)
        model.material("ElasticIsotropic", 1, 100.0, 0.3, density=1.0)
        L  = 5
        nz = 1
        nx = 1
        ny = 1

        nn = int((nz+1)*(nx+1)*(ny+1))

        # mesh generation
        model.block3D(*(nx, ny, nz), 1, 1, element, 1, {
                    1: [-1.0, -1.0,  0.0],
                    2: [ 1.0, -1.0,  0.0],
                    3: [ 1.0,  1.0,  0.0],
                    4: [-1.0,  1.0,  0.0],
                    5: [-1.0, -1.0,   L ],
                    6: [ 1.0, -1.0,   L ],
                    7: [ 1.0,  1.0,   L ],
                    8: [-1.0,  1.0,   L ]})
        model.fixZ(0.0, (1, 1, 1))
        return model
    return f


def CreateQ4(element, rotation=False):
    def f():
        print("Element: ", element)
        model = xara.Model(ndm=2, ndf=3 if rotation else 2)
        model.material("ElasticIsotropic", 1, 100.0, 0.3, density=1.0)
        section = xara.PlaneSection("PlaneStress", 1, 2.0)
        model.section(section)
        L  = 1
        ny = 1
        nx = 1
        model.node(1, (0.0, 0.0))
        model.node(2, ( L,  0.0))
        model.node(3, ( L,   L ))
        model.node(4, (0.0,  L ))

        if rotation:
            model.fix(1, (1, 1, 1))
            model.fix(2, (0, 1, 1))
            model.fix(3, (0, 0, 1))
            model.fix(4, (0, 0, 1))
        else:
            model.fix(1, (1, 1))
            model.fix(2, (0, 1))

        model.element(element, 1, (1, 2, 3, 4), section=section)

        return model
    return f


def CreateQ4_Pores(element):
    def f():
        #  iNode? jNode? kNode? lNode? thk? matTag? bulk? rho? perm_x? perm_y?
        print("Element: ", element)
        model = xara.Model(ndm=2, ndf=3)
        mat = 1
        model.material("ElasticIsotropic", mat, 100.0, 0.3, density=1.0)
        bulk = 1.0
        rho  = 0.0
        perm_x = 1.0
        perm_y = 1.0
        thick = 2.0
        void_stab = ()
        if "SSP" in element:
            thick, mat = mat, thick
            void_stab = 0.1, 0.1

        L  = 1
        model.node(1, (0.0, 0.0))
        model.node(2, ( L,  0.0))
        model.node(3, ( L,   L ))
        model.node(4, (0.0,  L ))

        model.fix(1, (1, 1, 1))
        model.fix(2, (0, 1, 1))
        model.fix(3, (0, 0, 1))
        model.fix(4, (0, 0, 1))

        model.element(element, 1, (1, 2, 3, 4), thick, mat, bulk, rho, perm_x, perm_y, *void_stab)

        return model
    return f


def CreateQ8(element):
    def f():
        print("Element: ", element)
        model = xara.Model(ndm=2, ndf=2)
        model.material("ElasticIsotropic", 1, 100.0, 0.3, density=1.0)
        section = xara.PlaneSection("PlaneStress", 1, 2.0)
        model.section(section)
        L  = 1
        ny = 1
        nx = 1
        model.node(1, (0.0, 0.0))
        model.node(2, ( L,  0.0))
        model.node(3, ( L,   L ))
        model.node(4, (0.0,  L ))
        model.node(5, (L/2, 0.0))
        model.node(6, (L, L/2))
        model.node(7, (L/2, L))
        model.node(8, (0.0, L/2))

        model.fix(1, (1, 1))
        model.fix(2, (0, 1))

        model.element(element, 1, tuple(range(1, 9)), section=section)

        return model
    return f


def model_nodes():
    element = "Brick02"
    print("Nodal Mass")

    model = xara.Model(ndm=3, ndf=3)
    model.material("ElasticIsotropic", 1, 100.0, 0.3, density=0.0)
    L  = 1
    nz = 1
    nx = 1
    ny = 1

    nn = int((nz+1)*(nx+1)*(ny+1))

    # mesh generation
    model.block3D(*(nx, ny, nz), 1, 1, element, 1, {
                  1: [-1.0, -1.0,  0.0],
                  2: [ 1.0, -1.0,  0.0],
                  3: [ 1.0,  1.0,  0.0],
                  4: [-1.0,  1.0,  0.0],
                  5: [-1.0, -1.0,   L ],
                  6: [ 1.0, -1.0,   L ],
                  7: [ 1.0,  1.0,   L ],
                  8: [-1.0,  1.0,   L ]})
    model.fixZ(0.0, (1, 1, 1))
    for node in model.getNodeTags():
        vol = (2*L)**3
        m = vol/8.0
        model.mass(node, (m,m,m))
    return model


def model_frame_3d():    
    pass


def _setup_modal(model):
    # Add modal damping to model, and return expected
    # damping matrix for comparison
    ev = model.eigen(3)#, solver="fullGenLapack")

    # Daming coefficients in first 3 modes
    c = [0.1, 0.02, 0.03]
    K = model.getTangent(k=1)
    M = model.getTangent(m=1)
    model.modalDamping(*c)

    # Compute eigenvectors and for damping matrix
    w, V = scipy.linalg.eigh(K, M, eigvals_only=False)
    D = np.zeros_like(K)
    for i in range(3):
        Q = M@V[:,i:i+1]
        D += c[i] * w[i] * (Q @ Q.T)
    return D




def check_damping(build_model, system="FullGeneral"):
    print("\n" + "="*60)
    print("System: ", system)
    aKinit = 0
    aM     = 0.13
    aKcurr = 0 #0.1
    aKcomm = 0 #0.0

    model = build_model()

    Dm = _setup_modal(model)

    model.rayleigh(aM, aKcurr, aKinit, aKcomm)


    model.timeSeries("Sine", 1, 0, 10, factor=1, period=0.1)
    dof = 1
    model.pattern("UniformExcitation", 1, dof, accel=1)


    gamma  = 0.5
    beta   = 0.25
    deltaT = 0.01
    c1 = 1.0
    c2 = gamma/(beta*deltaT)
    c3 = 1.0/(beta*deltaT*deltaT)

    model.integrator("Newmark", gamma, beta)
    # Keep two iterations to catch inconsistent modal tangents
    model.test("NormDispIncr", 1e-16, 2, 0)
    model.system(system)
    model.analysis("Transient")

    status = model.analyze(10, deltaT)
    assert status == 0


    Acomp = model.getTangent()

    K  = model.getTangent(k=1)
    M  = model.getTangent(m=1)
    D  = model.getTangent(c=1)

    assert not np.allclose(M, np.zeros_like(M), rtol=1e-6, atol=1e-15)

    Dexp = aKinit*K + aM*M + aKcurr*K + aKcomm*K

    # print("M: ", df(M))
    # print("K: ", df(K))
    # print("D(exp) = \n", df(Dexp))
    # print("D(comp) = \n", df(D))


    Aexp =  c1*K + c2*Dexp + c3*M
    model.wipe()
    assert np.allclose(Acomp, Aexp, rtol=1e-8, atol=1e-8)

    # TODO: check modal damping by invoking model.solveA()

    # TODO: check with different constraint handlers


def test():
    check_damping(model_nodes)
    check_damping(CreateH8("Brick02"))
    check_damping(CreateH8("stdBrick"))

    check_damping(CreateH8("stdBrick"), system="Umfpack02")
    check_damping(CreateH8("stdBrick"), system="Umfpack")

    check_damping(CreateH8("stdBrick"), system="BandGeneral")

    check_damping(CreateH8("stdBrick"), system="BandSPD")
    check_damping(CreateH8("SSPbrick"))
    check_damping(CreateH8("bbarBrick"))
    check_damping(CreateQ4("Q4", rotation=False))
    check_damping(CreateQ4("SSPquad"))
    check_damping(CreateQ8("Q8"))
    # check_damping(CreateQ4("bbarQuad"))
    check_damping(CreateQ4("enhancedQuad"))
    # check_damping(model_brick("H8E12"))

    check_damping(CreateQ4_Pores("quadUP"))
    check_damping(CreateQ4_Pores("SSPquadUP"))


if __name__ == "__main__":
    test()
