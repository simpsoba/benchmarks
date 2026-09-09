# Example from [1] 
#
# [1] C. M. Perez, 
#     “Nonlinear modeling of frame members for rapid infrastructure assessment,” 
#     Ph.D., University of California, Berkeley, Berkeley, CA, 2026.
#
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np


from xara.para import ChainModel
from xara.auto import StaticAnalysis
import pytest

def create_prism(length:    float,
                 element:   str,
                 boundary:  tuple,
                 steps = 1,
                 scale = 1.0,
                 orient: tuple  = (0, 0, 1),
                 transform: str = None,
                 divisions: int = 1,
                 rotation = None,
                 ndm=3):

    L  = length

    ne = divisions


    if ndm == 3:
        if rotation is not None:
            orient = tuple(map(float, rotation@orient))
    else:
        orient = ()

    nn = ne + 1
    def f(E, Iz):
        E   = E
        G   = 1.0
        A   = 2.0
        J   = 2.0
        Iy  = Iz
        Ay  = 2.0
        Az  = 2.0

        model = ChainModel(ndm=ndm, parameters=2)

        for i in range(1, nn+1):
            x = (i-1)/float(ne)*L
            if ndm == 3:
                location = (x, 0.0, 0.0)
            else:
                location = (x, 0.0)

            if rotation is not None:
                location = tuple(rotation@location)

            model.node(i, location)


        model.fix( 1, boundary[0])
        model.fix(nn, boundary[1])

        #
        # Define cross-section 
        #

        model.section("FrameElastic", 1,
                        E=E, A=A, Ay=Ay, Az=Az,
                        Iz=Iz, Iy=Iy, J=J, G=G)

        # Define geometric transformation
        model.geomTransf(transform, 1, *orient)

        # Define elements
        for i in range(1, ne+1):
            model.element(element, i, (i, i+1),
                        section=1,
                        shear=1,
                        transform=1)


        model.test("EnergyIncr", 1e-16, 10, 0)

        model.pattern("Plain", 1, "Linear", load={
            ne+1: [0, 0, 0] + [0, 0, -1]}
        )

        model.integrator("LoadControl", 2*np.pi*(E*Iz/length)*scale/steps)
        model.analysis("Static")
        model.system("Umfpack")
        return model

    return f




def solution(E, I, M, x, L):
    # Calculate the deflection at the free end of a cantilever beam
    # under a moment M at the free end.
    theta = M * x / (E*I)
    return (E*I/M)*np.sin(theta) - L # 
    #return E*I/M*(np.cos(theta) - 1.0)# +  (x - E*I/(E*I)) / (E * I)


# def gradient(E, I, M, dE=0, dI=0, dx=0, x=1.0):

#     dEI = dE*I + E*dI
#     theta = M * x / (E*I)
#     # dtheta = -M/(E*I)**2 * dEI * x
#     dtheta = (dEI * x - E * I * dx) / (E * I)

#     return (
#         dEI/M*np.sin(theta) + E*I/M*np.cos(theta) * dtheta
#     )


def run_bwd(element):
    L = 1.0
    ne = 20
    steps = 1

    builder = create_prism(
        length = L,
        element = element,
        transform = "Corotational04",
        boundary = ((1,1,1,1,1,1), 
                    (0,0,0,0,0,0)),
        steps = steps,
        divisions = ne
    )

    f = StaticAnalysis(builder, output=(ne+1, 1), steps=steps)

    E, I = 1.0, 2.0

    M = 2*np.pi*(E*I/L)

    print(f(E, I))

    df = jax.grad(f, (0,1))

    print(solution(E, I, M, L, L))

    dE, dIz = df(E,I)
    print("∂f/∂E =", dE)
    print("∂f/∂Iz =", dIz)

    print(jax.grad(solution, (0,1))(E, I, M, L, L))

    assert dE  == pytest.approx(-1.0, rel=1e-2)
    assert dIz == pytest.approx(-0.5, rel=1e-2)



    def _print(name, val, ref):
        if isinstance(name, tuple):
            name = f"∂{name[0]}/∂{name[1]}"

        print(name, "\t", val, "\t", ref, "\t", ((val-ref)/ref)*100, "%")
    
    _print(("u", "E"), dE, -1.0)
    _print(("u", "I"), dIz, -0.5)


def test_bwd():
    for element in ["ExactFrame", "ForceFrame", "PrismFrame"]:
        run_bwd(element)


if __name__ == "__main__":
    run_bwd("ExactFrame")
