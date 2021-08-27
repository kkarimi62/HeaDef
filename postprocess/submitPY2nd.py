if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 18,20 ]
	string=open('postprocess2nd.py').readlines() #--- python script
	#---
	PHI = np.logspace(np.log10(0.1),np.log10(0.9),8)
	nphi = len(PHI)
	#---
	for iphi in range( nphi ):
		#---	
		inums = lnums[ 0 ] - 1
		string[ inums ] = "\tjobname=\'HeaNiCoCrNatom10KTakeOneOutFreezeFract%sRlxd\'\n" % (iphi) #--- change job name
		#---	densities
		inums = lnums[ 1 ] - 1
		string[ inums ] = "\treadPath = os.getcwd() + \'/../lammpsRuns/HeaNiCoCrNatom10KTakeOneOutFreezeFract%sRlxd\'\n" % (iphi) #--- change job name

		sfile=open('junk%s.py'%iphi,'w');sfile.writelines(string);sfile.close()
		os.system( 'python3 junk%s.py'%iphi )
		os.system( 'rm junk%s.py'%iphi )
