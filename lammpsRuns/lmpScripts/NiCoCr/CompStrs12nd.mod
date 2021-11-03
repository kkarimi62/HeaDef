#--- compute stress
compute     peratom_${icel} freeGr stress/atom NULL
compute     p_${icel} freeGr reduce sum c_peratom_${icel}[1] c_peratom_${icel}[2] c_peratom_${icel}[3] c_peratom_${icel}[4] c_peratom_${icel}[5] c_peratom_${icel}[6]
#
variable    press equal -(c_p_${icel}[1]+c_p_${icel}[2]+c_p_${icel}[3])/(3*${volume})
variable 	pxx1_${dir}_${icel} equal -c_p_${icel}[1]/${volume}
variable 	pyy1_${dir}_${icel} equal -c_p_${icel}[2]/${volume}
variable 	pzz1_${dir}_${icel} equal -c_p_${icel}[3]/${volume}
variable 	pyz1_${dir}_${icel} equal -c_p_${icel}[6]/${volume}
variable 	pxz1_${dir}_${icel} equal -c_p_${icel}[5]/${volume}
variable 	pxy1_${dir}_${icel} equal -c_p_${icel}[4]/${volume}

#--- store initial stress
#thermo_style	custom	step	v_pxx0  v_press
#run	0

#variable	pxx1_${dir}_${icel}	equal ${pxx0}
#variable	pyy1_${dir}_${icel}	equal ${pyy0}
#variable	pzz1_${dir}_${icel}	equal ${pzz0}
#variable	pyz1_${dir}_${icel}	equal ${pyz0}
#variable	pxz1_${dir}_${icel}	equal ${pxz0}
#variable	pxy1_${dir}_${icel}	equal ${pxy0}

#uncompute	p
#uncompute	peratom_${icel}

