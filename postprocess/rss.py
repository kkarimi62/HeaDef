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
            9: '500',
            10:'600',
            11:'650',
            12:'700',
            13:'750',
            14:'800',
            15:'850',
            16:'900',
            17:'950',
		}

	nphi = len(PHI)
	#---
	count = 0
	for key in PHI:
		val = PHI[key]
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

		sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
		os.system( 'python3 junk%s.py'%count )
		os.system( 'rm junk%s.py'%count )
		count += 1
