def makeOAR( EXEC_DIR, node, core, tpartitionime, PYFIL, argv,argv2nd):
	#--- set environment variables
	sfile = open('.env','w')
	print('%s\n%s'%(argv,argv2nd),file=sfile)
	sfile.close()

	someFile = open( 'oarScript.sh', 'w' )
	print('#!/bin/bash\n',file=someFile)
	print('EXEC_DIR=%s\n'%( EXEC_DIR ),file=someFile)
#	print >> someFile, 'papermill --prepare-only %s/%s ./output.ipynb %s %s'%(EXEC_DIR,PYFIL,argv,argv2nd) #--- write notebook with a list of passed params
	print('jupyter nbconvert --execute $EXEC_DIR/%s --to html --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html'%(PYFIL), file=someFile)
	someFile.close()										  
#
if __name__ == '__main__':
	import os
#
	nruns	 = range(1)
	jobname  = {
					1:'NiCoCrNatom100KTemp600RhoFluc',
					2:'NiCoCrNatom100KTemp600RhoFlucRss', 
					3:'NiNatom100KTakeOneOut',
					4:'NiNatom100KReplaceCr',
					5:'NiNatom100KReplaceCo',
					6:'NiNatom1K',
					7:'NiCoCrNatom1000K', 
					8:'NiNatom1KEdgeDisl',
					9:'NiCoCrNatom1KT0Elastic',
					10:'FeNiT300Elasticity',
					11:'NiCoCrNatom100KTemp400sro',
				}[11]
	DeleteExistingFolder = False
	readPath = os.getcwd() + {
								1:'/../lammpsRuns/AmirData/shengAnnealed/Temp400', #--- source
							}[1]
	EXEC_DIR = '.'     #--- path for executable file
	durtn = '23:59:59'
	mem = '512gb'
	partition = ['cpu2019','bigmem','parallel','single'][1]
	argv = "path=%s"%(readPath) #--- don't change! 
	argv2nd = "indx=7\ntemperature=400\nload=500" 
	PYFILdic = { 
		0:'pressFluc.ipynb',
		1:'partition.ipynb',
		2:'Moduli.ipynb',
		3:'pressFluc2nd.ipynb',
		}
	keyno = 3
#---
#---
	PYFIL = PYFILdic[ keyno ] 
	#--- update argV
	#---
	if DeleteExistingFolder:
		os.system( 'rm -rf %s' % jobname ) # --- rm existing
	# --- loop for submitting multiple jobs
	counter = init = 0
	for counter in nruns:
		init = counter
		print(' i = %s' % counter)
		writPath = os.getcwd() + '/%s/Run%s' % ( jobname, counter ) # --- curr. dir
		os.system( 'mkdir -p %s' % ( writPath ) ) # --- create folder
		os.system( 'cp LammpsPostProcess*.py OvitosCna.py utility*.py %s' % ( writPath ) ) #--- cp python module
		makeOAR( writPath, 1, 1, durtn, PYFIL, argv+"/Run%s"%init, argv2nd) # --- make oar script
		os.system( 'chmod +x oarScript.sh; mv oarScript.sh .env %s; cp %s/%s %s' % ( writPath, EXEC_DIR, PYFIL, writPath ) ) # --- create folder & mv oar scrip & cp executable
		os.system( 'sbatch --partition=%s --mem=%s --time=%s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
						    --chdir %s -c %s -n %s %s/oarScript.sh'\
						   % ( partition, mem, durtn, jobname, counter, jobname, counter, jobname, counter \
						       , writPath, 1, 1, writPath ) ) # --- runs oarScript.sh!
											 

