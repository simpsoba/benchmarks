# Check that solvers dont fail when asked to solve size-zero systems.
#
import xara

def test():
    for system in ["ProfileSPD", "Umfpack", "BandSPD", "BandGen", "FullGen"]:
        ops = xara.Model(ndm=2, ndf=3)
        L = 100
        E = 29000
        A = 20
        I = 1400

        ops.node(1, (0,0))
        ops.node(2, (L,0))
        ops.fix(2,(1,1,1))
        ops.fix(1,(1,1,1))

        ops.geomTransf('Linear',1)
        ops.element('elasticBeamColumn',1,(1,2),A,E,I,1)

        ops.pattern('Plain',1,"Constant")
        # Unit displacement at node 2, DOF 2
        ops.sp(2,2,1.0, pattern=1)

        ops.constraints('Transformation')
        ops.system(system)

        ops.analysis('Static')
        assert ops.analyze(1) == 0
        ops.reactions()
