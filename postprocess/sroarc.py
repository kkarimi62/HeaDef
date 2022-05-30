if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 29, 33, 40 ]
	string=open('postprocess.py').readlines() #--- python script
	#---
	PHI={
	0:400, 
#	1:600, 
	2:800, 
	3:1000, 
	4:1200, 
	5:1400,
	}
	nphi = len(PHI)
	#---
	count = 0
	for key in PHI:
		val = PHI[key]
		#---	
		inums = lnums[ 0 ] - 1
#		string[ inums ] = "\t2:\'NiCoCrNatom100KTemp%s\',\n" % (PHI[iphi]) #--- change job name
		string[ inums ] = "\t11:\'NiCoCrNatom100KTemp%ssro\',\n" % (val) #--- change job name
		#---	
		inums = lnums[ 1 ] - 1
		string[ inums ] = "\t1:\'/../lammpsRuns/AmirData/shengAnnealed/Temp%s\',\n" % (val) #--- change job name
		#---	densities
		inums = lnums[ 2 ] - 1
		string[ inums ] = "\targv2nd = \'indx=%s\\ntemperature=%s\\nload=%s\'\n"%(7,val,500)

		sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
		os.system( 'python3 junk%s.py'%count )
		os.system( 'rm junk%s.py'%count )
		count += 1
