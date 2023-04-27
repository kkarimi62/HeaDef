if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	string=open('lammpsRuns2nd.py').readlines() #--- python script
	lnums = [ 40, 98] 
#	string=open('lammpsRuns-ncbj.py').readlines() #--- python script
#	lnums = [ 42, 103] #, 106   ]
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
	py = 'python'
	#---
	count = 0
	for keys_t in Temps:
					temp = Temps[keys_t]
					#--- write to
					inums = lnums[ 0 ] - 1
					string[ inums ] = "\t11:\'%sNatom100KMultipleTempIrradiatedAnneal/dpa2/temp%s\',\n"%(alloy,keys_t) #--- change job name

					inums = lnums[ 1 ] - 1
					string[ inums ] = "\t4:\' -var T %s -var t_sw 20.0 -var DataFile data_irradiated.dat -var nevery 100 -var ParseData 1 -var WriteData swapped.dat -var DUMP_FILE swapped.dump',\n"%(temp)					

#					inums = lnums[ 2 ] - 1
#					string[ inums ] = "\t7:\' -var buff 0.0 -var buffy 0.0 -var T %s -var P 0.0 -var nevery 1000 -var ParseData 1 -var DataFile data_minimized.txt -var DumpFile dumpThermalized.xyz -var WriteData Equilibrated.dat\',\n"%(temp)

			#---	read from
#					inums = lnums[ 1 ] - 1
#					string[ inums ] = "\t\'3\':\'/../simulations/%sNatom10KTemp300KMultipleRates/Rate%s\',\n"%(alloy,keys_r)

			#
					sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
					os.system( '%s junk%s.py'%(py,count ))
					os.system( 'rm junk%s.py'%count )
					count += 1
