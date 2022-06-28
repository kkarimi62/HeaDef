if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 26, 37, 8 ]
	string=open('postprocessNcbj.py').readlines() #--- python script
	#---
	PHI={

0:400,
1:600,
2:800,
3:1000,
4:1200,
5:1400

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
#12:500,

#0:450,
#1:500,
#2:1300,
#3:1200,
#4:1100,
#5:900,
#6:1000,
#7:800,
#8:750,
#9:400,
#10:550,
#11:700,
#12:650,
#13:600
	}
	nphi = len(PHI)
	#---
	count = 0
	for key in PHI:
		val = PHI[key]
		#---	
		inums = lnums[ 0 ] - 1
		string[ inums ] = "\t0:\'NiCoCrNatom100KDistortions/Temp%s\',\n" % (val) #--- change job name
		#---	
		inums = lnums[ 1 ] - 1
		string[ inums ] = "\t3:\'/../lammpsRuns/AmirData/shengAnnealed/Temp%s\',\n" % (val) #--- change job name

		#---	densities
		inums = lnums[ 2 ] - 1
		string[ inums ] = "\tconfParser.set(\'parameters\',\'temperature\',\'%s\')\n"%(val)

#		inums = lnums[ 2 ] - 1
#		string[ inums ] = "\tconfParser.set(\'parameters\',\'load\',\'%s\')\n"%(val)

		sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
		os.system( 'python3 junk%s.py'%count )
		os.system( 'rm junk%s.py'%count )
		count += 1
