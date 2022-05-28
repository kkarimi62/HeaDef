if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 22, 27, 33 ]
	string=open('postprocessNcbj.py').readlines() #--- python script
	#---
	PHI={
	-3:400,
	-1:450,
	0:500, 
	-2:550, 
	1:600, 
	2:650, 
	3:700, 
	4:750, 
#	5:800,
#	6:850,
#	7:900,
#	8:950,
	}
	nphi = len(PHI)
	#---
	count = 0
	for key in PHI:
		val = PHI[key]
		#---	
		inums = lnums[ 0 ] - 1
#		string[ inums ] = "\t2:\'NiCoCrNatom100KTemp%s\',\n" % (PHI[iphi]) #--- change job name
		string[ inums ] = "\t3:\'NiCoCrNatom100Krss/dislocated/load%s\',\n" % (val) #--- change job name
		#---	
		inums = lnums[ 1 ] - 1
		string[ inums ] = "\t1:\'/../lammpsRuns/AmirData/shengRss/Temp%s/dislocated\',\n" % (600) #--- change job name
		#---	densities
		inums = lnums[ 2 ] - 1
#		string[ inums ] = "\targv2nd = \'indx=7\\ntemperature=%s\'\n"%(PHI[iphi])
		string[ inums ] = "\targv2nd = \'indx=%s\\ntemperature=%s\\nload=%s\'\n"%(10,600,PHI[key])

		sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
		os.system( 'python3 junk%s.py'%count )
		os.system( 'rm junk%s.py'%count )
		count += 1
