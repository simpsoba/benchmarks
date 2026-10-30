# ===------------------------------------------------------------------=== #
# Conventional Triaxial Compression Test Using One Element              #
#
# University of Washington, Department of Civil and Environmental Eng      #
# Computational Geotechnics Eng Group, A. Ghofrani, P. Arduino - Dec 2013  #
#
# Basic units are m, Ton(metric), s                                        #
# ===------------------------------------------------------------------=== #
#
# https://opensees.berkeley.edu/wiki/index.php/J2CyclicBoundingSurface_Material
#
set strains {0.05}

set E 20000.0
set nu 0.499
set G [expr $E / 2.0 / (1 + $nu)]
set K [expr $E / 3.0 / (1 - 2.0 * $nu)]
set rho 1.7

set R [expr 100.0]
set su [expr sqrt(3.0 / 8.0) * $R]


for {set ii 0} {$ii < [llength $strains]} {incr ii} {
  wipe

  #
  # build model
  #
  model BasicBuilder -ndm 3 -ndf 3

  # create the nodes
  node 1  0.00000000 0.00000000 0.0
  node 2  0.00000000 1.00000000 0.0 
  node 3  1.00000000 1.00000000 0.0 
  node 4  1.00000000 0.00000000 0.0 
  node 5  0.00000000 0.00000000 1.0 
  node 6  0.00000000 1.00000000 1.0 
  node 7  1.00000000 1.00000000 1.0 
  node 8  1.00000000 0.00000000 1.0 


  # create the fixities
  fix 1   1  1  1 
  fix 2   1  0  1
  fix 3   0  0  1
  fix 4   0  1  1
  fix 5   1  1  0 
  fix 6   1  0  0
  fix 7   0  0  0
  fix 8   0  1  0

  # create the materials

  # nDMaterial ElasticIsotropic 1 100000 0.3 
  # nDMaterial J2CyclicBoundingSurface  tag? G? K? su? rho? h? m? h0? chi? beta? in kpa
  nDMaterial J2CyclicBoundingSurface 1 $G $K $su $rho $G 1.0 0.2 0.0 0.5

  # create the elements
  element SSPbrick    1    1 4 3 2 5 8 7 6   1


  # recorders
  recorder Node     -file "out/x-displacement.out"  -nodeRange 1 8 -dof 1 2 3 disp
  recorder Node     -file "out/x-velocity.out"      -nodeRange 1 8 -dof 1 2 3 vel
  recorder Node     -file "out/x-reactions.out"     -nodeRange 1 8 -dof 1 2 3 reaction
  recorder Element  -file "out/x-stress.out"     -ele 1  stress
  recorder Element  -file "out/x-strain.out"     -ele 1  strain

  # load pattern
  pattern Plain 1 {Series -time {0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -values {0.0 1.0 0.0 -1.0 0.0 1.0 0.0 -1.0 0.0} -factor -1.0} {
    sp 5 3 [lindex $strains $ii]
    sp 6 3 [lindex $strains $ii]
    sp 7 3 [lindex $strains $ii]
    sp 8 3 [lindex $strains $ii]
  }

  # analysis
  constraints Transformation
  test        NormDispIncr 1e-9 50 0
  algorithm   Newton
  numberer    Plain
  system      BandSPD; #SparseSPD
  integrator  LoadControl 0.004
  analysis    Static
   
  analyze    2000

  reactions
  verify value [nodeDisp 5 3]  0.00000000000002189360 1e-12
  verify value [nodeDisp 2 2] -0.00000448309979891026 1e-12

  verify error [nodeReaction 1 3] 22.41549891367408164911 1e-12
  wipe
}
