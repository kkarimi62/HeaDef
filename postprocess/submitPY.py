if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 21,26,32 ]
	string=open('postprocessNcbj.py').readlines() #--- python script
	#---
	PHI = [400,600,800,1000,1200,1400]
	nphi = len(PHI)
	#---
	for iphi in range( nphi ):
		#---	
		inums = lnums[ 0 ] - 1
		string[ inums ] = "\t2:\'NiCoCrNatom100KTemp%s\',\n" % (PHI[iphi]) #--- change job name
		#---	
		inums = lnums[ 1 ] - 1
		string[ inums ] = "\t1:\'/../lammpsRuns/AmirData/shengAnnealed/Temp%s\',\n" % (PHI[iphi]) #--- change job name
		#---	densities
		inums = lnums[ 2 ] - 1
		string[ inums ] = "\targv2nd = \'indx=7\\ntemperature=%s\'\n"%(PHI[iphi])

		sfile=open('junk%s.py'%iphi,'w');sfile.writelines(string);sfile.close()
		os.system( 'python3 junk%s.py'%iphi )
		os.system( 'rm junk%s.py'%iphi )
