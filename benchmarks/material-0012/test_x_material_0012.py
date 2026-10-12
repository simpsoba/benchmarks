# Orthotropic Wrapper

# This example demonstrates the `OrthotropicWrapper` material. 
#
# This example is adapted from [this](https://opensees.github.io/OpenSeesDocumentation/user/manual/material/ndMaterials/Orthotropic.html) web page. 
#
# A simple example which evaluates the Yield domain in the plane-stress plane (Szz = 0) of the original isotropic J2Plasticity model (Sy = 400 MPa) and of its orthotropic counter-part (Sx = 1.5*Sy, Ex = 1.5*Ey).
#

import math

E = 200000.0
v = 0.3
G = E/(2.0*(1.0+v))
K = E/(3.0*(1.0-2.0*v))
Fy = 400.0


import xara

base = xara.MultiaxialMaterial("PlasticJ2", E=E, G=G, Fy=Fy)

orth = xara.MultiaxialMaterial("Orthotropic", 
                               base, 
                               Ex=E*1.5, 
                               Ey=E, 
                               Ez=E,
                               Gxy=G, Gyz=G, Gzx=G,
                               vxy=v, vyz=v, vzx=v,
                               Axx=1.0/1.5, 
                               Ayy=1.0, Azz=1.0,
                               Axyxy=1.0, 
                               Ayzyz=1.0, Axzxz=1.0)

def create_patch(material):

    model = xara.Model(ndm=2, ndf=2)

    # Add both materials to the model
    model.material(base)
    model.material(orth)

    assert isinstance(base.asdict(), dict)
    assert isinstance(orth.asdict(), dict)

    # Create a plane section
    section = xara.PlaneSection("PlaneStress", material, 1.0)

    # Add the section to the model
    model.section(section)

    # Create three nodes for the triangle
    model.node( 1, 0, 0 )
    model.node( 2, 1, 0 )
    model.node( 3, 0, 1 )

    # Fix boundary conditions
    model.fix( 1,   1, 1)
    model.fix( 2,   0, 1)
    model.fix( 3,   1, 0)

    # Create the triangle element
    model.element("tri31", 1,  (1, 2, 3),  section=section)

    return model


def analyze_dir(dX, dY, material):

    # the 2D model
    model = create_patch(material)

    # a simple ramp
    model.timeSeries("Linear", 1, factor=2.0*Fy)

    # imposed stresses
    model.pattern("Plain", 1, 1 )
    model.load( 2, dX, 0.0 )
    model.load( 3, 0.0, dY )

    # analyze
    model.constraints("Transformation")
    model.numberer("Plain")
    model.system("FullGeneral" )
    model.test("NormDispIncr", 1.0e-6, 3, 9)
    model.algorithm("Newton")

    dLambda = 0.1
    dLambdaMin = 0.001
    Lambda = 0.0
    sX = 0.0
    sY = 0.0
    while 1 :
        model.integrator( "LoadControl", dLambda )
        model.analysis( "Static" )
        ok = model.analyze(1)
        if ok == 0:
            stress = model.eleResponse(1, "material", 1, "stress" )
            sX = stress[0]
            sY = stress[1]
            Lambda += dLambda
            if Lambda > 0.9999:
                break
        else:
            dLambda /= 2.0
            if dLambda < dLambdaMin:
                break

    # done
    return sX, sY



def analyze_surface():
    NDiv = 48
    NP = NDiv+1
    dAngle = 2.0*math.pi/NDiv
    SX = [0.0]*NP
    SY = [0.0]*NP
    SXortho = [0.0]*NP
    SYortho = [0.0]*NP
    for i in range(NDiv):
        angle = i*dAngle
        dX = math.cos(angle)
        dY = math.sin(angle)
        iso = analyze_dir(dX, dY, base)
        ortho = analyze_dir(dX, dY, orth)
        SX[i] = iso[0]
        SY[i] = iso[1]
        SXortho[i] = ortho[0]
        SYortho[i] = ortho[1]
    SX[-1] = SX[0]
    SY[-1] = SY[0]
    SXortho[-1] = SXortho[0]
    SYortho[-1] = SYortho[0]

    return (SX, SY, SXortho, SYortho)


def test():
    SX, SY, SXortho, SYortho = analyze_surface()


    assert  460 > abs(min(SX)) > 400
    assert  460 > abs(max(SX)) > 400
    assert  460 > abs(min(SY)) > 400
    assert  460 > abs(max(SY)) > 400

    assert  690 > abs(min(SXortho)) > 600
    assert  690 > abs(max(SXortho)) > 600
    assert  465 > abs(min(SYortho)) > 400
    assert  465 > abs(max(SYortho)) > 400


if __name__ == "__main__":
    test()
