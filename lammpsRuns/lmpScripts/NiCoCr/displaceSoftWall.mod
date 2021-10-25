# NOTE: This script should not need to be
# modified. See in.elastic for more info.
#
# Find which reference length to use

if "${dir} == 1" then &
   "variable len0 equal ${lx0}" 
if "${dir} == 2" then &
   "variable len0 equal ${ly0}" 
if "${dir} == 3" then &
   "variable len0 equal ${lz0}" 
if "${dir} == 4" then &
   "variable len0 equal ${lz0}" 
if "${dir} == 5" then &
   "variable len0 equal ${lz0}" 
if "${dir} == 6" then &
   "variable len0 equal ${ly0}" 

# Reset box and simulation parameters

clear
box tilt large
read_restart restart.equil
include ${INC}/potential.mod

#if "${dir} == 1" then &
#	"dump        1 all custom 1 dump.xyz id type x y z" 
#if "${dir} == 1" then &
#	"dump_modify 1 flush yes append yes"
#run	1

# Negative deformation

variable delta equal -${up}*${len0}
variable deltaxy equal -${up}*xy
variable deltaxz equal -${up}*xz
variable deltayz equal -${up}*yz
if "${dir} == 1" then &
   "change_box all x delta 0 ${delta} xy delta ${deltaxy} xz delta ${deltaxz} remap units box"
if "${dir} == 2" then &
   "change_box all y delta 0 ${delta} yz delta ${deltayz} remap units box"
if "${dir} == 3" then &
   "change_box all z delta 0 ${delta} remap units box"
if "${dir} == 4" then &
   "change_box all yz delta ${delta} remap units box"
if "${dir} == 5" then &
   "change_box all xz delta ${delta} remap units box"
if "${dir} == 6" then &
   "change_box all xy delta ${delta} remap units box"

#--- compute stress
#compute     peratom freeGr stress/atom NULL
#compute     p freeGr reduce sum c_peratom[1] c_peratom[2] c_peratom[3] c_peratom[4] c_peratom[5] c_peratom[6]
#
#variable    press1 equal -(c_p[1]+c_p[2]+c_p[3])/(3*${volume})
#variable 	pxx1 equal -c_p[1]/${volume}
#variable 	pyy1 equal -c_p[2]/${volume}
#variable 	pzz1 equal -c_p[3]/${volume}
#variable 	pyz1 equal -c_p[6]/${volume}
#variable 	pxz1 equal -c_p[5]/${volume}
#variable 	pxy1 equal -c_p[4]/${volume}

#--- set dynamics 
#fix         set_force frozGr setforce 0.0 0.0 0.0
#velocity    frozGr set 0.0 0.0 0.0 #--- set velocity

# Relax atoms positions

minimize ${etol} ${ftol} ${maxiter} ${maxeval}


# Obtain new stress tensor

#thermo_style	custom	step v_pxx1	v_sxx0
#run	0
 
#variable tmp equal pxx
#variable pxx1 equal ${tmp}
#variable tmp equal pyy
#variable pyy1 equal ${tmp}
#variable tmp equal pzz
#variable pzz1 equal ${tmp}
#variable tmp equal pxy
#variable pxy1 equal ${tmp}
#variable tmp equal pxz
#variable pxz1 equal ${tmp}
#variable tmp equal pyz
#variable pyz1 equal ${tmp}


# Compute elastic constant from pressure tensor

#variable C1neg equal ${d1}
#variable C2neg equal ${d2}
#variable C3neg equal ${d3}
#variable C4neg equal ${d4}
#variable C5neg equal ${d5}
#variable C6neg equal ${d6}

# Reset box and simulation parameters

#clear
#box tilt large
#read_restart restart.equil
#include ${INC}/potential.mod

#if "${dir} == 1" then &
#	"dump        1 all custom 1 dump.xyz id type x y z" 
#if "${dir} == 1" then &
#	"dump_modify 1 flush yes append yes"
#run	1

# Positive deformation

#variable delta equal ${up}*${len0}
#variable deltaxy equal ${up}*xy
#variable deltaxz equal ${up}*xz
#variable deltayz equal ${up}*yz
#if "${dir} == 1" then &
#   "change_box all x delta 0 ${delta} xy delta ${deltaxy} xz delta ${deltaxz} remap units box"
#if "${dir} == 2" then &
#   "change_box all y delta 0 ${delta} yz delta ${deltayz} remap units box"
#if "${dir} == 3" then &
#   "change_box all z delta 0 ${delta} remap units box"
#if "${dir} == 4" then &
#   "change_box all yz delta ${delta} remap units box"
#if "${dir} == 5" then &
#   "change_box all xz delta ${delta} remap units box"
#if "${dir} == 6" then &
#   "change_box all xy delta ${delta} remap units box"

#--- compute stress
#compute     peratom freeGr stress/atom NULL
#compute     p freeGr reduce sum c_peratom[1] c_peratom[2] c_peratom[3] c_peratom[4] c_peratom[5] c_peratom[6]
#variable    press1 equal -(c_p[1]+c_p[2]+c_p[3])/(3*${volume})
#variable 	pxx1 equal -c_p[1]/${volume}
#variable 	pyy1 equal -c_p[2]/${volume}
#variable 	pzz1 equal -c_p[3]/${volume}
#variable 	pyz1 equal -c_p[6]/${volume}
#variable 	pxz1 equal -c_p[5]/${volume}
#variable 	pxy1 equal -c_p[4]/${volume}

# Relax atoms positions

#--- set dynamics 
#fix         set_force frozGr setforce 0.0 0.0 0.0
#velocity    frozGr set 0.0 0.0 0.0 #--- set velocity

#minimize ${etol} ${ftol} ${maxiter} ${maxeval}

#if "${dir} == 1" then &
#	"undump	1"

# Obtain new stress tensor
#thermo_style	custom	step v_pxx1	v_sxx0
#run	0
 
#variable tmp equal pe
#variable e1 equal ${tmp}
#variable tmp equal press
#variable p1 equal ${tmp}
#
#
#variable tmp equal pxx
#variable pxx1 equal ${tmp}
#variable tmp equal pyy
#variable pyy1 equal ${tmp}
#variable tmp equal pzz
#variable pzz1 equal ${tmp}
#variable tmp equal pxy
#variable pxy1 equal ${tmp}
#variable tmp equal pxz
#variable pxz1 equal ${tmp}
#variable tmp equal pyz
#variable pyz1 equal ${tmp}


# Compute elastic constant from pressure tensor

#variable C1pos equal ${d1}
#variable C2pos equal ${d2}
#variable C3pos equal ${d3}
#variable C4pos equal ${d4}
#variable C5pos equal ${d5}
#variable C6pos equal ${d6}

# Combine positive and negative 

#variable C1${dir} equal 0.5*(${C1neg}+${C1pos})
#variable C2${dir} equal 0.5*(${C2neg}+${C2pos})
#variable C3${dir} equal 0.5*(${C3neg}+${C3pos})
#variable C4${dir} equal 0.5*(${C4neg}+${C4pos})
#variable C5${dir} equal 0.5*(${C5neg}+${C5pos})
#variable C6${dir} equal 0.5*(${C6neg}+${C6pos})

# Delete dir to make sure it is not reused

#variable dir delete

write_restart restart.deform${dir}

