import sys
from pathlib import Path
import numpy as np
from numpy import cos,sin,sqrt,pi,exp
import xara

# Linear Elastic SINGLE DOF Model Transient Analysis

# REFERENCES
#
# 1) Chopra, A.K. "Dynamics of Structures: Theory and Applications"
# Prentice Hall, 1995.
#   - Sections 3.1, Section 3.2 and Section 6.4



print("sdofTransient.tcl: Verification of Elastic SDOF systems (Chopra)")

#
# global variables
#
gravity = 386.4
testOK  = 0     # variable used to keep track of SUCCESS or FAILURE
tol     = 1.0e-4

# procedure to build a linear model
#   input args: K - desired stiffness
#               periodStruct - desired structure period (used to compute mass)
#               dampRatio (zeta) - desired damping ratio

def elastic_sdof(K, period, dampRatio):

    wn = 2.0 * pi / period
    m  = K/(wn**2)

    model = xara.Model(ndm=1,  ndf=1)

    model.node( 1,  0.)
    model.node( 2,  0., mass=m)

    model.uniaxialMaterial("Elastic",  1, K)
    model.element("ZeroLength", 1, *[1, 2], mat=1,  dir=1)

    model.fix(1, 1)

    # add damping using rayleigh damping on the mass term
    a0 = 2.0*wn*dampRatio
    model.rayleigh(a0, 0., 0., 0.)

    return model


def build_analysis(model, integrator):
    model.constraints('Plain')
    model.numberer('Plain')
    model.algorithm('Newton')
    model.system('ProfileSPD')
    model.integrator(*integrator)
    model.analysis('Transient')
    return model


def harmonic_undamped(tCurrent, w, wn, P, K):

    return P/K * 1.0/(1 - (w*w)/(wn*wn)) * (sin(w*tCurrent) - (w/wn)*sin(wn*tCurrent))


def harmonic_damped(t, w, wn, dampRatio, P, K):
    wd = (wn*sqrt(1-dampRatio*dampRatio))
    wwn2 = (w*w)/(wn*wn)
    det = (1.0-wwn2)*(1-wwn2) + 4.0 * dampRatio*dampRatio*wwn2
    ust = P/K

    C = ust/det * (1.-wwn2)
    D = ust/det * (-2.*dampRatio*w/wn)
    A = -D
    B = ust/det * (1.0/wd) * ((-2. * dampRatio * w/wn) - w * (1.0 - wwn2))

    return exp(-dampRatio*wn*t)*(A * cos(wd * t) + B*sin(wd*t)) + C*sin(w*t) + D*cos(w*t)


def test_earthquake():
    #
    # Section 6.4 - Earthquake Response of Linear System
    #
    print("\n\nEarthquake Response (Section 6.4)\n")

    testOK = 0

    periodStruct = 0.8
    K  = 2.0

    tol = 3e-2
    dt  = 0.01 # analysis time step

    dir = Path(__file__, "..").resolve()

    dir = Path(__file__, "..").resolve()
    dt = 0.02
    plain_file = str(dir/"elCentro.txt")
    accel = np.loadtxt(plain_file)

    # print table header
    print("%15s%15s%15s%15s"%('Period', 'Damping', 'OpenSees', 'Reference'))

    # perform analysis for various periods and damping ratios
    for  period, dampRatio, peak  in {(0.5, 0.02, 2.67 ),
                                      (1.0, 0.02, 5.97 ),
                                      (2.0, 0.02, 7.47 ),
                                      (2.0, 0.00, 9.91 ),
                                      (2.0, 0.02, 7.47 ),
                                      (2.0, 0.05, 5.37 )}:


        # Create a model
        model = elastic_sdof(K, period, dampRatio)

        # add load pattern
#       model.timeSeries('Path', 1, filePath='elCentro.dat', dt=0.02,  factor=gravity)
        model.timeSeries('Path', 1, values=accel, dt=dt,  factor=gravity)
        model.pattern("UniformExcitation", 1, 1, accel=1)
#       model.pattern("Plain", 1, 1, factor=-1/m)

        # Configure analysis
        model = build_analysis(model, ["Newmark", 0.5 , 1.0/6.0])

        maxU = 0.0
        for _ in range(int(len(accel.data)*0.02/dt)):
            model.analyze(1, dt)
            u = abs(model.nodeDisp(2))
            if u > maxU:
                maxU = u


        print("%15.2f%15.2f%15.2f%15.2f"%(period,dampRatio,maxU,peak))


        if abs(maxU-peak) > tol :
            testOK = -1;
            print(f"FAILED earthquake response ({period = }, {dampRatio =  })")
            print(f"    {maxU} {peak}, {abs(peak-maxU) = } > {tol}")

    assert testOK == 0



if __name__ == "__main__":

    #
    # Section 3.1 - Harmonic Vibrartion of Undamped Elastic SDOF System
    #
    print("Undamped System Harmonic Exciatation (Section 3.1)")

    # model properties
    periodStruct = 0.8
    K  = 2.0
    # harmonic loading
    P = 2.0
    periodForce = 5.0
    tFinal = 2.251*periodForce

    dt = periodStruct/1.0e4; # something, small, for, accuracy

    # derived quantaties
    w  =  2.0 * pi / periodForce
    wn =  2.0 * pi / periodStruct

    #
    # build the model
    #
    model = elastic_sdof(K, periodStruct, 0.0)

    # add load pattern
    time = np.linspace(0, 100*periodForce, 100)
    series = 1
    model.timeSeries('Trig', series, 0.0, 100.0*periodForce, periodForce, factor=P)
    #   model.timeSeries("Path", series, dt=dt, values=list(np.sin(time)*P)) #, factor=P)

    model.pattern("Plain",  1,  series, load={2: [1.0]})

    # build analysis
    model = build_analysis(model, ["Newmark", 0.5, 1.0/6.0])

    # perform analysis, checking at every step
    tCurrent = 0.
    while tCurrent < tFinal :
        model.analyze(1, dt)
        tCurrent  = model.getTime()
        uOpenSees = model.nodeDisp(2)
        uExact    = harmonic_undamped(tCurrent, w, wn, P, K)

        if abs(uExact - uOpenSees) > tol :
            testOK = -1;
            print(f"failed  undamped harmonic: {abs(uExact-uOpenSees) = } > {tol} at time {tCurrent}")
            tCurrent = tFinal

    print("%20s%15.5f%10s%15.5f" % ("OpenSees: ", uOpenSees, "Exact: ", uExact))



    #
    # Section 3.2 - Harmonic Vibrartion of Damped Elastic SDOF System
    #
    print("\n\nDamped System Harmonic Excitation (Section 3.2)")

    tol = 1.0e-2
    dampRatio = 0.05

    # build the model
    model = elastic_sdof(K, periodStruct, dampRatio)

    # add load pattern
    model.timeSeries('Trig', 1, 0.0, 100.0*periodForce, periodForce , factor=P)
    model.pattern("Plain", 1, 1, load={2: [1.0]})

    # build analysis
    model = build_analysis(model, ["Newmark", 0.5, 1.0/6.0])

    t = 0.
    while t < tFinal :
        model.analyze(1, dt)
        t = model.getTime()
        uOpenSees = model.nodeDisp(2)
        uExact    = harmonic_damped(t, w, wn, dampRatio, P, K)
        if abs(uExact-uOpenSees) > tol:
            testOK = -1
            print(f"  failed  damped harmonic>, {abs(uExact-uOpenSees) = } > {tol} at time {t}")
            t = tFinal


    print(f"\n  Displacement Comparison at {tCurrent} (sec):")
    print("%20s%15.5f%10s%15.5f"%("OpenSees: ", uOpenSees, "Exact: ", uExact))

    test_earthquake()

