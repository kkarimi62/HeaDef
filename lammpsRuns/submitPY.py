if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 37, 58, 94 ]
	string=open('lammpsRuns2nd.py').readlines() #--- python script
	#---
	PHI = [23.1142383485014, 11.5571191742507, 7.704746116167133, 5.136497410778088, 3.302034049785914] #range(1331) #np.logspace(np.log10(0.1),np.log10(0.9),8)
	nphi = len(PHI)
	#---
	for iphi in range( nphi ):
		#---	
		inums = lnums[ 0 ] - 1
#		string[ inums ] = "\tjobname=\'NiCoCrNatom1KT0Elastic2nd%s\'\n" % (iphi) #--- change job name
		string[ inums ] = "\t8:\'NiCoCrNatom10KT0Elastic%s\',\n" % (iphi) #--- change job name
		#---	densities
		inums = lnums[ 1 ] - 1
		string[ inums ] = "\t5:[\'data_init.txt\',\'ScriptGroup.%s.txt\'],\n" % (iphi) #--- change job name
#		string[ inums ] = "\tsourcePath = os.getcwd() +\'/../postprocess/HeaNiCoCrNatom10KTakeOneOutFreezeFract%s\'\n" % (iphi) #--- change job name
		#---	densities
		inums = lnums[ 2 ] - 1
		string[ inums ] = "\t\'p0\':\' data_init.txt %s"%PHI[iphi]+" %s\'%(os.getcwd()+\'/../postprocess\'),\n" #--- change job name
		
		sfile=open('junk%s.py'%iphi,'w');sfile.writelines(string);sfile.close()
		os.system( 'python junk%s.py'%iphi )
		os.system( 'rm junk%s.py'%iphi )
