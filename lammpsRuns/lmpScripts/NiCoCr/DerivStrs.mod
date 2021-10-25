# These formulas define the derivatives w.r.t. strain components
# Constants uses $, variables use v_

variable Sxx0	equal	v_sxx0_${icel}
variable sxx0	equal	${Sxx0}
#
variable Syy0	equal	v_syy0_${icel}
variable syy0	equal	${Syy0}
#
variable Szz0	equal	v_szz0_${icel}
variable szz0	equal	${Szz0}
#
variable Syz0	equal	v_syz0_${icel}
variable syz0	equal	${Syz0}
#
variable Sxz0	equal	v_sxz0_${icel}
variable sxz0	equal	${Sxz0}
#
variable Sxy0	equal	v_sxy0_${icel}
variable sxy0	equal	${Sxy0}


variable d1 equal -(v_pxx1-v_sxx0)/(v_delta/v_len0)*${cfac}
variable d2 equal -(v_pyy1-v_syy0)/(v_delta/v_len0)*${cfac}
variable d3 equal -(v_pzz1-v_szz0)/(v_delta/v_len0)*${cfac}
variable d4 equal -(v_pyz1-v_syz0)/(v_delta/v_len0)*${cfac}
variable d5 equal -(v_pxz1-v_sxz0)/(v_delta/v_len0)*${cfac}
variable d6 equal -(v_pxy1-v_sxy0)/(v_delta/v_len0)*${cfac}


variable C1${dir}_${icel} equal ${d1}
variable C2${dir}_${icel} equal ${d2}
variable C3${dir}_${icel} equal ${d3}
variable C4${dir}_${icel} equal ${d4}
variable C5${dir}_${icel} equal ${d5}
variable C6${dir}_${icel} equal ${d6}

