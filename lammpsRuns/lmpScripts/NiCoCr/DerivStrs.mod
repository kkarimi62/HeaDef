# These formulas define the derivatives w.r.t. strain components
# Constants uses $, variables use v_

variable Sxx0	equal	v_sxx0_${icel}
variable sxx0	equal	${Sxx0}
variable Sxx1	equal	v_pxx1_${dir}_${icel}
variable sxx1	equal	${Sxx1}
#
variable Syy0	equal	v_syy0_${icel}
variable syy0	equal	${Syy0}
variable Syy1	equal	v_pyy1_${dir}_${icel}
variable syy1	equal	${Syy1}
#
variable Szz0	equal	v_szz0_${icel}
variable szz0	equal	${Szz0}
variable Szz1	equal	v_pzz1_${dir}_${icel}
variable szz1	equal	${Szz1}
#
variable Syz0	equal	v_syz0_${icel}
variable syz0	equal	${Syz0}
variable Syz1	equal	v_pyz1_${dir}_${icel}
variable syz1	equal	${Syz1}
#
variable Sxz0	equal	v_sxz0_${icel}
variable sxz0	equal	${Sxz0}
variable Sxz1	equal	v_pxz1_${dir}_${icel}
variable sxz1	equal	${Sxz1}
#
variable Sxy0	equal	v_sxy0_${icel}
variable sxy0	equal	${Sxy0}
variable Sxy1	equal	v_pxy1_${dir}_${icel}
variable sxy1	equal	${Sxy1}


variable d1 equal -(v_sxx1-v_sxx0)/(v_delta/v_len0)*${cfac}
variable d2 equal -(v_syy1-v_syy0)/(v_delta/v_len0)*${cfac}
variable d3 equal -(v_szz1-v_szz0)/(v_delta/v_len0)*${cfac}
variable d4 equal -(v_syz1-v_syz0)/(v_delta/v_len0)*${cfac}
variable d5 equal -(v_sxz1-v_sxz0)/(v_delta/v_len0)*${cfac}
variable d6 equal -(v_sxy1-v_sxy0)/(v_delta/v_len0)*${cfac}


variable C1${dir}_${icel} equal ${d1}
variable C2${dir}_${icel} equal ${d2}
variable C3${dir}_${icel} equal ${d3}
variable C4${dir}_${icel} equal ${d4}
variable C5${dir}_${icel} equal ${d5}
variable C6${dir}_${icel} equal ${d6}

