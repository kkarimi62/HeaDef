print "id type x y z C11 C12 C13 C14 C15 C16 C22 C23 C24 C25 C26 C33 C34 C35 C36 C44 C45 C46 C55 C56 C66 sxx syy szz syz sxz sxy" file "ElasticConst.txt"
#print "#icel C55 C56 C66" file "ElasticConst.txt"
variable 	icel 	loop	${natom} 
	label loop2
	variable	C11all	equal	v_C11_${icel}
	variable	C12all	equal	v_C12_${icel}
	variable	C13all	equal	v_C13_${icel}
	variable	C14all	equal	v_C14_${icel}
	variable	C15all	equal	v_C15_${icel}
	variable	C16all	equal	v_C16_${icel}
	#
	variable	C22all	equal	v_C22_${icel}
	variable	C23all	equal	v_C23_${icel}
	variable	C24all	equal	v_C24_${icel}
	variable	C25all	equal	v_C25_${icel}
	variable	C26all	equal	v_C26_${icel}
	#
	variable	C33all	equal	v_C33_${icel}
	variable	C34all	equal	v_C34_${icel}
	variable	C35all	equal	v_C35_${icel}
	variable	C36all	equal	v_C36_${icel}
	#
	variable	C44all	equal	v_C44_${icel}
	variable	C45all	equal	v_C45_${icel}
	variable	C46all	equal	v_C46_${icel}
	#
	variable	C55all	equal	v_C55_${icel}
	variable	C56all	equal	v_C56_${icel}
	#
	variable	C66all	equal	v_C66_${icel}
	#

	variable	virxx	equal	-v_sxx0_iatom${icel}
	variable	viryy	equal	-v_syy0_iatom${icel}
	variable	virzz	equal	-v_szz0_iatom${icel}
	variable	viryz	equal	-v_syz0_iatom${icel}
	variable	virxz	equal	-v_sxz0_iatom${icel}
	variable	virxy	equal	-v_sxy0_iatom${icel}
	#
	variable	xx		equal	x[${icel}]		
	variable	yy		equal	y[${icel}]		
	variable	zz		equal	z[${icel}]		
	#
	variable	idd		equal	id[${icel}]		
	variable	itype		equal	type[${icel}]		
	#
	print "${idd} ${itype} ${xx} ${yy} ${zz} ${C11all} ${C12all} ${C13all} ${C14all} ${C15all} ${C16all} ${C22all} ${C23all} ${C24all} ${C25all} ${C26all} ${C33all} ${C34all} ${C35all} ${C36all} ${C44all} ${C45all} ${C46all} ${C55all} ${C56all} ${C66all} ${virxx} ${viryy} ${virzz} ${viryz} ${virxz} ${virxy}" append "ElasticConst.txt"
#	print "${idd} ${C66all}" append "ElasticConst.txt"
next    icel
jump    ${INC}/print.mod  loop2
#--- end of outer loop

