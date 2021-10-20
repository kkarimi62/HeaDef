# NOTE: This script can be modified for different atomic structures, 
# units, etc. See in.elastic for more info.
#

# Define the finite deformation size. Try several values of this
# variable to verify that results do not depend on it.
variable up equal 1.0e-6
 
# Define the amount of random jiggle for atoms
# This prevents atoms from staying on saddle points
variable atomjiggle equal 1.0e-5

# Uncomment one of these blocks, depending on what units
# you are using in LAMMPS and for output

# metal units, elastic constants in eV/A^3
#units		metal
#variable cfac equal 6.2414e-7
#variable cunits string eV/A^3

# metal units, elastic constants in GPa
units		metal
variable cfac equal 1.0e-4
variable cunits string GPa

# real units, elastic constants in GPa
#units		real
#variable cfac equal 1.01325e-4
#variable cunits string GPa

# Define minimization parameters
variable etol equal 0.0 
variable ftol equal 1.0e-10
variable maxiter equal 100
variable maxeval equal 1000
variable dmax equal 1.0e-2

# generate the box and atom positions using a diamond lattice
#variable a equal 5.43

#boundary	p p p

#lattice         diamond $a
#region		box prism 0 2.0 0 3.0 0 4.0 0.0 0.0 0.0
#create_box	1 box
#create_atoms	1 box

#--- define variables
variable 	a      			equal   ${cutoff}   #--- lattice constant
variable    volume          equal   ${natoms}*${a}^3.0/4.0 #--- natom * vol. of the voronoi cell
variable    lx              equal   floor(${volume}^(1.0/3.0)/${a})

# ---------- Create Atoms ---------------------
## define crystal structure and lattice constant a0
## define direction vectors, i.e., set x=[100], y=[010], z=[001] and origin point.
#
lattice    fcc ${a} orient    x 1 0 0 orient y 0 1 0 orient z 0 0 1 &   
           origin 0.1 0.1 0.1
region    mybox block 0 ${lx} 0 ${lx} 0 ${lx}   ## define box sizes along x, y, z (in the unit of a0)
create_box      3 mybox              ## create the simulation box, allowing a max of three species
create_atoms    1 box               ## create type-1 metal atoms in the box

change_box	all	triclinic

# Need to set mass to something, just to satisfy LAMMPS
mass 1 1.0e-20

