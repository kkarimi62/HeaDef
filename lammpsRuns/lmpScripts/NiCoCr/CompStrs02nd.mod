#--- compute stress
compute     peratom all stress/atom NULL
#compute     p freeGr reduce sum c_peratom[1] c_peratom[2] c_peratom[3] c_peratom[4] c_peratom[5] c_peratom[6]
#
#variable    press equal -(c_peratom[1]+c_peratom[2]+c_peratom[3])/(3*${volume})
variable 	pxx0 vector -c_peratom[1]
variable 	pyy0 vector -c_peratom[2]
variable 	pzz0 vector -c_peratom[3]
variable 	pyz0 vector -c_peratom[6]
variable 	pxz0 vector -c_peratom[5]
variable 	pxy0 vector -c_peratom[4]

#--- store initial stress
thermo_style	custom	step	v_pxx0	pxx v_press
run	0	post	no

#variable	sxx0_${icel}	equal ${pxx0}
#variable	syy0_${icel}	equal ${pyy0}
#variable	szz0_${icel}	equal ${pzz0}
#variable	syz0_${icel}	equal ${pyz0}
#variable	sxz0_${icel}	equal ${pxz0}
#variable	sxy0_${icel}	equal ${pxy0}

#uncompute	p
#uncompute	peratom


