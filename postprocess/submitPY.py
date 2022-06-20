if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 31, 8, 9 ]
	string=open('postprocessNcbj.py').readlines() #--- python script
	#---
	PHI={
	-1:400,
#	-2:450,
	0:500, 
	-3:550, 
	1:600, 
	2:650, 
	3:700, 
	4:750, 
#	5:800,
#	6:850,
#	7:900,
#	8:950,
#	9:1000,
#	10:1100,
#	11:1200,
#	12:1300,
#	13:1400,
	}
	nphi = len(PHI)
	#---
	count = 0
	for key in PHI:
		val = PHI[key]
		#---	
		inums = lnums[ 0 ] - 1
		string[ inums ] = "\t4:\'NiCoCrNatom100KTemp600Rss/dislocated/load%s\',\n" % (val) #--- change job name
		#---	
#		inums = lnums[ 1 ] - 1
#		string[ inums ] = "\t1:\'/../lammpsRuns/AmirData/shengAnnealed/Temp%s\',\n" % (PHI[iphi]) #--- change job name
		#---	densities
		inums = lnums[ 1 ] - 1
		string[ inums ] = "\tconfParser.set(\'parameters\',\'temperature\',\'%s\')\n"%(600)

		inums = lnums[ 2 ] - 1
		string[ inums ] = "\tconfParser.set(\'parameters\',\'load\',\'%s\')\n"%(val)

		sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
		os.system( 'python3 junk%s.py'%count )
		os.system( 'rm junk%s.py'%count )
		count += 1
