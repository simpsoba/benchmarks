#--------------------------------------------------------
# Conventional triaxial compression test
#
# File is generated for the purposes of testing the#
# Drucker-Prager model 
##
#   Created:  03.16.2009 CRM#
#   Updated:  12.02.2011 CRM#
##
##
#--------------------------------------------------------

#--define material parameters for the model
#---bulk modulus
set k       27777.78
#---shear modulus
set G       9259.26
#---yield stress
set Fy    5.0
#---failure surface and associativity
set rho     0.398
set rhoBar  0.398
#---isotropic hardening
set Fs      0.0
set Fo      0.0
set delta1  0.0
#---kinematic hardening
set H       0.0
set theta   1.0
#---tension softening
set delta2  0.0
#---mass density
set mDen    1.7

#-------------------------------------------------------
# create the modelBuilder and build the model
#-------------------------------------------------------
model BasicBuilder -ndm 3 -ndf 3

#--create the nodes
node 4 0.0 0.0 0.0 ; # 1
node 1 1.0 0.0 0.0 ; # 2
node 2 1.0 1.0 0.0 ; # 3
node 3 0.0 1.0 0.0 ; # 4
node 8 0.0 0.0 1.0 ; # 5
node 5 1.0 0.0 1.0 ; # 6
node 6 1.0 1.0 1.0 ; # 7
node 7 0.0 1.0 1.0 ; # 8

#--triaxial test boundary conditions
fix 1 0 1 1
fix 2 0 0 1
fix 3 1 0 1
fix 4 1 1 1
fix 5 0 1 0
fix 6 0 0 0
fix 7 1 0 0
fix 8 1 1 0


#--material models
nDMaterial DruckerPrager 2 \
   -K      27777.78  \
   -G       9259.26  \
   -Fy         5.0   \
   -Rvol       0.398 \
   -Rbar       0.398 \
   -Fs         0.0   \
   -Fo         0.0   \
   -Hsat       0.0   \
   -H          0.0   \
   -theta      1.0   \
   -delta2     0.0   \
   -density    1.7

#--create the element
#         type  tag      nodes        matID   bforce1  bforce2  bforce3
element stdBrick 1    1 2 3 4 5 6 7 8   2      0.0      0.0      0.0

#-------------------------------------------------------
# create the recorders
#-------------------------------------------------------
set step 0.1


#-------------------------------------------------------
# create the loading
#-------------------------------------------------------

#--pressure magnitude
set p 10.0
set pNode [expr -$p*0.25] 

#--loading: hydrostatic pressure
pattern Plain 1 {Series -time {0 10 100} -values {0 1 1} -factor 1} { 
    load 1  $pNode    0.0 0.0
    load 2  $pNode $pNode 0.0
    load 3     0.0 $pNode 0.0
    load 5  $pNode    0.0 0.0
    load 6  $pNode $pNode 0.0
    load 7     0.0 $pNode 0.0
}

#--loading: deviator stress
pattern Plain 2 {Series -time {0 10 100} -values {0 1 5} -factor 1} { 
    load 5  0.0 0.0 $pNode
    load 6  0.0 0.0 $pNode
    load 7  0.0 0.0 $pNode
    load 8  0.0 0.0 $pNode
}

#-------------------------------------------------------
# create the analysis
#-------------------------------------------------------

integrator LoadControl 0.1
numberer RCM
system Umfpack ; # SparseGeneral
constraints Transformation
test NormDispIncr 1e-5 1 9; #10 1
algorithm Newton
analysis Static

analyze 1000
#verify value [analyze 963] 0

verify value "[getTime]" 96.49999999999879

