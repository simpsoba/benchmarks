#
# https://openseesdigital.com/2026/04/20/pathological-convergence/
#
import xara
from math import isclose


def test_ops_2D():
    ft = 1
    kip = 1

    inch = ft/12
    ksi = kip/inch**2

    L = 20*ft
    E = 29000*ksi
    I = 800*inch**4
    A = 15*inch**2

    w = 1.0*kip/ft

    ops = xara.Model(ndm=2, ndf=3)

    ops.node(1,0,0)
    ops.node(2,L,0)

    ops.fix(1,1,1,1)
    ops.fix(2,0,1,0)

    ops.geomTransf("Linear", 1)

    Np = 3
    ops.section('Elastic',1,E,A,I)
    ops.beamIntegration('Lobatto',1,1,Np)

    ops.element('forceBeamColumn',1,1,2,1,1)

    ops.pattern('Plain',1,"Constant")
    ops.eleLoad('-ele',1,'-type','beamUniform',-w)

    # ops.test('NormDispIncr',1e-8,10,1) # No equilibrium, assertion fails
    ops.test('NormUnbalance',1e-8,10,1) # Equilibrium, assertion passes
    ops.analysis('Static')

    assert ops.analyze(1) == 0
    ops.reactions()

    assert isclose(ops.nodeDisp(2,3),(w*L**3)/(48*E*I))
