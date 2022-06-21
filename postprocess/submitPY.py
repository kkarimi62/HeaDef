if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 29, 8, 9 ]
	string=open('postprocessNcbj.py').readlines() #--- python script
	#---
	PHI={
#0:1300,
#1:1200,
#2:1100,
#3:1000,
#4:850,
#5:950,
#6:900,
#7:800,
#8:750,
#9:700,
#10:650,
#11:600,
12:500,
	}
	nphi = len(PHI)
	#---
	count = 0
	for key in PHI:
		val = PHI[key]
		#---	
		inums = lnums[ 0 ] - 1
		string[ inums ] = "\t4:\'NiCoCrNatom100KTemp600/dislocated/load%s\',\n" % (val) #--- change job name
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
