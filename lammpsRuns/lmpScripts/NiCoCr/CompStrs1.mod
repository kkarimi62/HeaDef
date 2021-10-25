#--- compute stress
compute     peratom freeGr stress/atom NULL
compute     p freeGr reduce sum c_peratom[1] c_peratom[2] c_peratom[3] c_peratom[4] c_peratom[5] c_peratom[6]
#
variable    press equal -(c_p[1]+c_p[2]+c_p[3])/(3*${volume})
variable 	pxx0 equal -c_p[1]/${volume}
variable 	pyy0 equal -c_p[2]/${volume}
variable 	pzz0 equal -c_p[3]/${volume}
variable 	pyz0 equal -c_p[6]/${volume}
variable 	pxz0 equal -c_p[5]/${volume}
variable 	pxy0 equal -c_p[4]/${volume}

#--- store initial stress
thermo_style	custom	step	v_pxx0	pxx v_press
run	0

variable	pxx1	equal ${pxx0}
variable	pyy1	equal ${pyy0}
variable	pzz1	equal ${pzz0}
variable	pyz1	equal ${pyz0}
variable	pxz1	equal ${pxz0}
variable	pxy1	equal ${pxy0}

uncompute	p
uncompute	peratom

