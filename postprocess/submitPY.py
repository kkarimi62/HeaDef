if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 22, 33 ]
	string=open('postprocessNcbj.py').readlines() #--- python script
	#---
#	PHI = [400,600,800,1000,1200,1400]
	PHI={
            9:'500',
            10:'600',
            11:'700',
            12:'800',
            13:'900',
		}

	nphi = len(PHI)
	#---
	for key, val in PHI:
		#---	
		inums = lnums[ 0 ] - 1
#		string[ inums ] = "\t2:\'NiCoCrNatom100KTemp%s\',\n" % (PHI[iphi]) #--- change job name
		string[ inums ] = "\t3:\'NiCoCrNatom100KTemp600/dislocated/load%s\',\n" % (val) #--- change job name
		#---	
#		inums = lnums[ 1 ] - 1
#		string[ inums ] = "\t1:\'/../lammpsRuns/AmirData/shengAnnealed/Temp%s\',\n" % (PHI[iphi]) #--- change job name
		#---	densities
		inums = lnums[ 1 ] - 1
#		string[ inums ] = "\targv2nd = \'indx=7\\ntemperature=%s\'\n"%(PHI[iphi])
		string[ inums ] = "\targv2nd = \'indx=%s\\ntemperature=600\'\n"%(key)

		sfile=open('junk%s.py'%iphi,'w');sfile.writelines(string);sfile.close()
		os.system( 'python3 junk%s.py'%iphi )
		os.system( 'rm junk%s.py'%iphi )
