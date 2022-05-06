def makeOAR( EXEC_DIR, node, core, partitionime, PYFIL, argv,argv2nd):
	#--- set environment variables
	sfile = open('.env','w')
	print('%s\n%s'%(argv,argv2nd),file=sfile)
	sfile.close()
	#---
	someFile = open( 'oarScript.sh', 'w' )
	print('#!/bin/bash\n', file=someFile)
	print('EXEC_DIR=%s\n' %( EXEC_DIR ), file=someFile)
	print('module load python-jupyter/dev-x86_64-gcc46-python35\n',file=someFile)
	print('jupyter nbconvert --execute $EXEC_DIR/%s --to html --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html'%(PYFIL), file=someFile)
	someFile.close()										  
#
if __name__ == '__main__':
	import os

	nruns	 = range(1)
	jobname  = ['NiCoCrNatom100KTakeOneOutRlxd',
				'NiCoCrNatom200KTemp600Annealed', 
				'NiCoCrNatom100KTemp1200'
				][2]
	DeleteExistingFolder = False
	readPath = os.getcwd() + ['/../testRuns/glassCo5Cr2Fe40Mn27Ni26',
								'/../lammpsRuns/AmirData/shengAnnealed/Temp1200',
							][1] # --- source
	EXEC_DIR = '.'     #--- path for executable file
	durtn = '02:59:59'
	resources = {'mem':'16gb', 'partition':['o12h','a12h','i12h'][2],'nodes':1,'ppn':1}
	argv = "path=%s"%(readPath) #--- don't change! 
	argv2nd = "indx=7\ntemperature=1200" 
	PYFILdic = { 
		0:'pressFluc2nd.ipynb',
		}
	keyno = 0
#---
#---
	PYFIL = PYFILdic[ keyno ] 
	#--- update argV
	#---
	os.system( 'rm -rf %s' % jobname ) # --- rm existing
	# --- loop for submitting multiple jobs
	for counter in nruns:
		print(' i = %s' % counter)
		writPath = os.getcwd() + '/%s/Run%s' % ( jobname, counter ) # --- curr. dir
		os.system( 'mkdir -p %s' % ( writPath ) ) # --- create folder
		os.system( 'cp LammpsPostProcess.py OvitosCna.py utility.py %s' % ( writPath ) ) #--- cp python module
		makeOAR( writPath, 1, 1, durtn, PYFIL, argv+"/Run%s"%counter, argv2nd) # --- make oar script
		os.system( 'chmod +x oarScript.sh; mv oarScript.sh .env %s; cp %s/%s %s' % ( writPath, EXEC_DIR, PYFIL, writPath ) ) # --- create folder & mv oar scrip & cp executable
		os.system( 'qsub -q %s -l nodes=%s:ppn=%s -l walltime=%s -N %s.%s -o %s -e %s -d %s  %s/oarScript.sh'\
			%( resources['partition'], resources['nodes'], resources['ppn'], durtn, jobname, counter, writPath, writPath, writPath , writPath ) ) # --- runs oarScript.sh!
											 

