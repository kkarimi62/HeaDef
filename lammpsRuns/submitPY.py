if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	string=open('lammpsRuns2nd.py').readlines() #--- python script
	lnums = [ 40, 98 ]
	#---

	Temps  = {
				0:300,
				1:400,
				2:500,
				3:600,
				4:700,
				5:800,
				6:900,
				7:1000,
			}


	alloy = 'nicocr'
	
	#---
	count = 0
	for keys_t in Temps:
		temp = Temps[keys_t]
			#--- write to
					inums = lnums[ 0 ] - 1
					string[ inums ] = "\t\'3\':\'%snicocrNatom100KMultipleTempIrradiatedAnneal/dpa2/temp%s\',\n"%(alloy,keys_t) #--- change job name

					inums = lnums[ 0 ] - 2
					string[ inums ] = "\t4:\' -var T %s -var t_sw 20.0 -var DataFile data_irradiated.dat -var nevery 100 -var ParseData 1 -var WriteData swapped.dat',\n"%(temp)					

			#---	read from
#					inums = lnums[ 1 ] - 1
#					string[ inums ] = "\t\'3\':\'/../simulations/%sNatom10KTemp300KMultipleRates/Rate%s\',\n"%(alloy,keys_r)

			#
					sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
					os.system( 'python3 junk%s.py'%count )
					os.system( 'rm junk%s.py'%count )
					count += 1
