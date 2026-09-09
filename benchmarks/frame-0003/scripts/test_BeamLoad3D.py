#
# Simply supported beam with distributed loads
#
import xara

def printRow(quantity, expected, computed):
  print("%10s %10f %10f" % (quantity, expected, computed))
  # assert abs(expected - computed) < 1e-5*abs(expected), f"{quantity} does not match: {expected} vs {computed}"


def test_eleLoad_3D():
  #
  E =  30000.0
  A =  20.0
  Iz = 1400.0
  Iy = 1200.0
  G =  18000.0
  J =  Iz+Iy

  L =  100.0
  beta = 0.2
  lp = (beta*L)

  # Point loads
  Py = 200.0
  Pz = 250.0
  N  = 150.0
  aL = 0.6
  a  = aL*L

  # Distributed loads
  wy = 3.0
  wz = 4.0
  wx = 2.0

  nIP = 5

  elements = [
    "PrismFrame",
    "ForceFrame",
    "forceBeamColumn"
  ]

  for element in elements:
    print(f"Element: {element}")

    model = xara.Model('basic', ndm=3,  ndf=6)

    model.node(1, (0.0, 0.0, 0.0))
    model.node(2, ( L , 0.0, 0.0))

    model.fix(1, (1, 1, 1, 1, 0, 0))
    model.fix(2, (0, 1, 1, 0, 0, 0))

    sec = 1
    model.section('Elastic', 1, E, A, Iz, Iy, G, J)

    model.geomTransf("Linear", 1, (0, 0, 1))

    # Create elements
    tag   = 1
    nodes = 1, 2

    model.element(element, tag, nodes, section=sec, transform=1)#, shear=0)

    model.pattern("Plain", 1, "Constant")
    model.eleLoad("-ele", 1,  "-type", "-beamPoint", Py/2,  Pz/2, aL, N/2)
    model.eleLoad("-ele", 1,  "-type", "-beamPoint", Py/2,  Pz/2, aL, N/2)


    model.eleLoad ("-ele", 1,  "-type", "-beamUniform", wy/2,  wz/2, wx/2)
    model.eleLoad ("-ele", 1,  "-type", "-beamUniform", wy/2,  wz/2, wx/2)


    niter = 2
    model.test("NormUnbalance", 1.0e-10, niter)
    model.algorithm("Newton")
    model.integrator("LoadControl", 1.0)
    model.constraints("Plain")
    model.system("ProfileSPD")
    model.numberer("Plain")
    model.analysis("Static")

    assert model.analyze(2) == 0


    d1 = aL*N*L/(E*A)
    d2 = (wx*L**2/(2*E*A))

    printRow("axial", (d1+d2), model.nodeDisp( 2, 1))

    Vy = (Py*(1-aL))
    Vz = (Pz*(1-aL))


    d1 = (-Vy/(6*E*Iz*L)*(a*a*L-2*a*L*L))
    d2 = (wy*L**3/(24*E*Iz))

    printRow("theta_zi", (d1+d2), model.nodeDisp( 1, 6))

    d1 = (-Vy/(6*E*Iz*L)*(a*a*L+a*L*L))

    printRow("theta_zi", (d1-d2), model.nodeDisp( 2, 6))


    d1 = (-Vz/(6*E*Iy*L)*(a*a*L-2*a*L*L))
    d2 = (wz*L**3/(24*E*Iy))

    printRow("theta_yi", (d1+d2), -model.nodeDisp( 1, 5))

    d1 = -Vz/(6*E*Iy*L)*(a*a*L+a*L*L)

    printRow("theta_yj", d1-d2, -model.nodeDisp(2, 5))

    print("")


if __name__ == "__main__":
  test_eleLoad_3D()
