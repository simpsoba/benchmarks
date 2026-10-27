from opensees import openseespy as os
import math
from math import sqrt

E = 200000.0
v = 0.3
G = E/(2.0*(1.0+v))
K = E/(3.0*(1.0-2.0*v))
Fy  = 400.0
Fxx = 600.0

def analyze_openseespy(dX, dY, type):

    # info
    print("Analyze direction (%g, %g)" % (dX, dY))

    # the 2D model
    os.wipe()
    os.model( "basic", "-ndm", 2, "-ndf", 2 )

    # the material
    os.nDMaterial( "J2Plasticity", 1, K, G, Fy, Fy, 0.0, 0.0)

    # the orthotropic wrapper
    if type == "ortho":
        Ex = E*1.5
        Ey = E
        Ez = E
        Gxy = G
        Gyz = G
        Gzx = G
        vxy = v
        vyz = v
        vzx = v
        Asigmaxx = 1.0/1.5 # fx_iso/fx_ortho
        # nDMaterial Orthotropic $tag $theIsoMat $Ex $Ey $Ez $Gxy $Gyz $Gzx $vxy $vyz $vzx $Asigmaxx $Asigmayy $Asigmazz $Asigmaxyxy $Asigmayzyz $Asigmaxzxz.
        os.nDMaterial( "Orthotropic", 2, 1, Ex, Ey, Ez, Gxy, Gyz, Gzx, vxy, vyz, vzx, Asigmaxx, 1.0, 1.0, 1.0, 1.0, 1.0)
        os.nDMaterial( "PlaneStress", 3, 2)

    # a triangle
    os.node( 1, 0, 0 )
    os.node( 2, 1, 0 )
    os.node( 3, 0, 1 )
    os.element("tri31", 1,   1, 2, 3,   1.0, "PlaneStress", 3 if type == "ortho" else 1 )

    # fixity
    os.fix( 1,   1, 1)
    os.fix( 2,   0, 1)
    os.fix( 3,   1, 0)

    # a simple ramp
    os.timeSeries( "Linear", 1, "-factor", 2.0*Fy )

    # imposed stresses
    os.pattern( "Plain", 1, 1 )
    os.load( 2, dX, 0.0 )
    os.load( 3, 0.0, dY )

    # analyze
    os.constraints( "Transformation" )
    os.numberer( "Plain" )
    os.system( "FullGeneral" )
    os.test( "NormDispIncr", 1.0e-6, 3, 0)
    os.algorithm( "Newton" )

    dLambda = 0.1
    dLambdaMin = 0.001
    Lambda = 0.0
    sX = 0.0
    sY = 0.0
    while True:
        os.integrator( "LoadControl", dLambda )
        os.analysis( "Static" )
        ok = os.analyze( 1 )
        if ok == 0:
            stress = os.eleResponse(1, "material", 1, "stress")
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


def test_openseespy():
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

        iso = analyze_openseespy(dX, dY, "iso")
        ortho = analyze_openseespy(dX, dY, "ortho")

        SX[i] = iso[0]
        SY[i] = iso[1]
        SXortho[i] = ortho[0]
        SYortho[i] = ortho[1]

    SX[-1] = SX[0]
    SY[-1] = SY[0]
    SXortho[-1] = SXortho[0]
    SYortho[-1] = SYortho[0]

    assert  460 > abs(min(SX)) > 400
    assert  460 > abs(max(SX)) > 400
    assert  460 > abs(min(SY)) > 400
    assert  460 > abs(max(SY)) > 400

    assert  690 > abs(min(SXortho)) > 600
    assert  690 > abs(max(SXortho)) > 600
    assert  465 > abs(min(SYortho)) > 400
    assert  465 > abs(max(SYortho)) > 400

    # assert Fy*1.1 > min(SY) > -Fy*1.1
    # assert Fxx*1.1 > min(SXortho) > -Fxx*1.1
    # assert Fy*1.1  > min(SYortho) > -Fy*1.1


if __name__ == "__main__":
    test_openseespy()
