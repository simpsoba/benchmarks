# This is a minimal example regarding the use of Section Aggregator with Fiber section and TwoNodeLink elements.

# Global Y axis is vertical. 
# Model is an upright cantilever. 
# Local y aligned with Global X.
import opensees.openseespy as ops
import numpy as np
import pandas as pd

inch = 1.0
ft = 12*inch
kip = 1.0
ksi = kip/(inch**2)

# Define pier geometry
pier_height = 79*inch # Length/height of element
pier_thick = 9*inch # Cross section thickness
pier_width = 9*inch # Cross section width

## DEFINE MATERIAL TAGS

tag_section_fiber = 701
tag_section_aggregated = 710

def fiber_section(tag_material, numFibers=(16, 4)): # Simple homogenous fiber element
    # define simple fiber elements
    colWidth = pier_width # 109 inch
    colHeight = pier_thick # 9 inch

    # Specify some arbitrary torsional stiffness
    ops.section('Fiber', tag_section_fiber, '-GJ', 1e8)
    # Create fibers
    botLeft = np.array([-colWidth/2, -colHeight/2])
    topRight = np.array([colWidth/2, colHeight/2])
    print(f"Making fiber section. Bottom left at {botLeft}. TopRight at {topRight}")
    ops.patch('rect', tag_material, *numFibers, *botLeft, *topRight)
    #create_patch_rect(ax, tag_material, *numFibers, botLeft, topRight, color='tab:pink')


def build_nodes():
    ## MODEL GEOMETRY
    ops.node(100, *(0, 0, 0))
    ops.node(101, *(0, pier_height, 0))

    # boundary conditions
    ops.fix(100, *(1, 1, 1), *(1, 1, 1))
        

def build_model(useSprings = False, element="twoNodeLinkSection"): 
    # generate rough mesh (6 nodes total). Number left to right, bottom to top
    print("Building Geometry")

    # set modelbuilder
    ops.model('basic', '-ndm', 3, '-ndf', 6)
    
    # Simple homogenous fiber
    
    tag_debug_mat = 360
    #ops.uniaxialMaterial('SteelMPF', tag_debug_p, fy, fy, Es, alpha, alpha, R0, A1, A2)
    E = 1000 # ksi
    ops.uniaxialMaterial('Elastic', tag_debug_mat, E) # ksi if using as a fiber material
    A = pier_width * pier_thick # in2
    

    tag_p_spring = 370
    tag_t_spring = 371
    tag_vy_spring = 372
    tag_vz_spring = 373
    tag_my_spring = 374
    tag_mz_spring = 375

    ops.uniaxialMaterial('Elastic', tag_p_spring, E * A / pier_height) # Same as fiber section
    ops.uniaxialMaterial('Elastic', tag_t_spring, 1e8)
    ops.uniaxialMaterial('Elastic', tag_vy_spring, 300)
    ops.uniaxialMaterial('Elastic', tag_vz_spring, 300)
    ops.uniaxialMaterial('Elastic', tag_my_spring, E * A * (pier_thick/2)**2 / 4) # Same as 2x2 fiber section
    ops.uniaxialMaterial('Elastic', tag_mz_spring, E * A * (pier_width/2)**2 / 4) # Same as 2x2 fiber section

    fiber_section(tag_debug_mat, numFibers=(2,2)) # tag_section_fiber. Define simple fiber section with a homogenous material

    if useSprings:
        # Aggregate with 6 independent springs
        ops.section('Aggregator', tag_section_aggregated, 
                    *[tag_vy_spring, 'Vy', tag_vz_spring, 'Vz',                                                      tag_my_spring, 'My', tag_mz_spring, 'Mz', tag_t_spring, 'T', tag_p_spring, 'P']) # Try with only springs
    else:
        # Aggreagte to a fiber section
        ops.section('Aggregator', tag_section_aggregated, 
                    *[tag_vy_spring, 'Vy', tag_vz_spring, 'Vz'], 
                    '-section', tag_section_fiber)


    # Build geometry
    build_nodes()

    # use two node link elements
    if element == "twoNodeLinkSection":
        ops.element('twoNodeLinkSection', 200, *(100, 101), tag_section_aggregated,
                    '-orient', *(0, 1, 0), *(1, 0, 0),
                    '-shearDist', *(0.1, 0.1)) # orient so local x is up (global y) and local y is along strong axis (global x) direction
    else:
        # Use disp-beam column element
        tag_transf = 800
        tag_integration = 900
        ops.geomTransf('Linear', tag_transf, *(0, 0, 1)) # Orient in the same way (local y aligned with global X)
        ops.beamIntegration('Lobatto', tag_integration, tag_section_aggregated, 2)
        ops.element('dispBeamColumn', 200, *(100, 101), tag_transf, tag_integration)




def get_stiffness():
    # View full stiffness matrix.
    ops.constraints('Transformation')
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.analysis('Transient')
    
    ops.integrator('GimmeMCK',0,0,1)
    ops.analyze(1,0.0)
    N = ops.systemSize() # Has to be done after analyze
    K = np.array(ops.printA('-ret')).reshape((N,N))
    print(f"Stiffness matrix of size {N}x{N}:")
    print(pd.DataFrame(K))
    return K


if __name__ == '__main__':
    # run the analysis
    print("Getting stiffness from fiber model")
    ops.wipe()
    build_model(useSprings=False)
    K_fiber = get_stiffness()
    print(f"K44 and K66 (bending dofs) for K_fiber: K44 = {K_fiber[3][3]:.1f}. K66 = {K_fiber[5][5]:.1f}\n")
    ops.printModel("-JSON", "-file", "model_fiber.json")


    print("Getting stiffness from spring only model")
    ops.wipe()
    build_model(useSprings=True)
    K_spring = get_stiffness()
    print(f"K44 and K66 (bending dofs) for K_spring: K44 = {K_spring[3][3]:.1f}. K66 = {K_spring[5][5]:.1f}\n")

    if False:
        print("Getting stiffness from fiber model")
        ops.wipe()
        build_model(useSprings=False, element="dispBeamColumn")
        K_fiber = get_stiffness()
        print(f"K44 and K66 (bending dofs) for K_fiber: K44 = {K_fiber[3][3]:.1f}. K66 = {K_fiber[5][5]:.1f}\n")


        print("Getting stiffness from spring only model")
        ops.wipe()
        build_model(useSprings=True, element="dispBeamColumn")
        K_spring = get_stiffness()
        print(f"K44 and K66 (bending dofs) for K_spring: K44 = {K_spring[3][3]:.1f}. K66 = {K_spring[5][5]:.1f}\n")



