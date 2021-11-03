#--- compute stress
compute     peratom_${icel} freeGr stress/atom NULL
compute     p_${icel} freeGr reduce sum c_peratom_${icel}[1] c_peratom_${icel}[2] c_peratom_${icel}[3] c_peratom_${icel}[4] c_peratom_${icel}[5] c_peratom_${icel}[6]
#
variable    press equal -(c_p_${icel}[1]+c_p_${icel}[2]+c_p_${icel}[3])/(3*${volume})
variable 	sxx0_${icel} equal -c_p_${icel}[1]/${volume}
variable 	syy0_${icel} equal -c_p_${icel}[2]/${volume}
variable 	szz0_${icel} equal -c_p_${icel}[3]/${volume}
variable 	syz0_${icel} equal -c_p_${icel}[6]/${volume}
variable 	sxz0_${icel} equal -c_p_${icel}[5]/${volume}

#variable	sxx0_${icel}	equal ${pxx0}
#variable	syy0_${icel}	equal ${pyy0}
#variable	szz0_${icel}	equal ${pzz0}
#variable	syz0_${icel}	equal ${pyz0}
#variable	sxz0_${icel}	equal ${pxz0}
#variable	sxy0_${icel}	equal ${pxy0}

