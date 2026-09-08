import xara
from xara.rotations import exp, log
import numpy as np
import pytest


def create_model(element, transform, R=None):
    model = xara.Model(ndm=3, ndf=6)
    L = 1.0
    if R is None:
        R = np.eye(3)

    model.node(1, 0.0, 0.0, 0.0)
    model.node(2,  *(R.T@[L , 0.0, 0.0]))

    model.section("ElasticFrame", 1, 
                  E=1,
                  G=1,
                  A=1,
                  Ay=1,
                  Az=1,
                  Iy=1,
                  Iz=1,
                  J=1)
    
    model.geomTransf(transform, 1, (0,0,1))
    model.element(element, 1, [1,2], section=1, transform=1)


    model.fix(1, (1, 1, 1, 1, 1, 1))
    model.fix(2, (1, 1, 1, 1, 1, 1))

    return model

def run(element, transform):
    print(transform)
    model = create_model(element, transform)
    model.analysis("Static")
    model.analyze(1)

    Ri = [[ 1.0, -0.5, 0.25],
          [-0.4,  0.7, 0.10]]
    for i in range(2):
        model.setNodeDisp(i+1, [0]*3 + Ri[i])

    # model.print()
    model.analysis("Static")
    model.analyze(1, operation="update")
    e1 = model.state.element(1).section(1).strain()[0:7]
    print(e1)


    R = exp([0.2, 1.2, -0.5])
    model = create_model(element, transform)
    model.analysis("Static")
    model.analyze(1)
    # for i in range(2):
    #     RI = log((R@exp(Ri[i]))).tolist()
    #     model.setNodeDisp(i+1, [0]*3 + RI)
    X0 = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]) # L = 1.0
    ]

    for i in range(2):
        # 1. Update the rotations
        RI = log((R @ exp(Ri[i]))).tolist()
        
        # 2. Update the translations (Displacement = R*X0 - X0)
        X_new = R @ X0[i]
        U_trans = (X_new - X0[i]).tolist()
        
        # 3. Apply both translation and rotation
        model.setNodeDisp(i+1, U_trans + RI)

    # model.print()
    model.analysis("Static")
    model.analyze(1, operation="update")
    e2 = model.state.element(1).section(1).strain()[0:7]
    print(e2)

    if element == "ExactFrame" or transform == "Identity":
        assert e1[3] == pytest.approx(-1.274645, abs=1e-5) # torsion
        assert e2[3] == pytest.approx(-1.263986, abs=1e-5) #
        assert e1[4] == pytest.approx( 1.267559, abs=1e-5) # torsion

    elif element in {"CosseratFrame", "CosseratFrame01"} and transform == "Spherical":
        assert e1[3] == pytest.approx(e2[3], rel=1e-12)
    else:
        raise ValueError()



def test_frame_1000_simo():
    run("ExactFrame", "Linear")
    run("ExactFrame", "Identity")
    run("ExactFrame", "Spherical")



def test_frame_1000_jelenic():
    run("CosseratFrame",   "Identity")
    run("CosseratFrame",   "Spherical")
    run("CosseratFrame01", "Spherical")


