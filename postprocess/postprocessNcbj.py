from backports import configparser

def makeOAR( EXEC_DIR, node, core, partitionime, PYFIL, argv):
	#--- parse conf. file
	confParser = configparser.ConfigParser()
	confParser.read('.env')
	#--- set parameters
	confParser.set('parameters','temperature','600.0')
	confParser.set('parameters','load','850')
	confParser.set('input files','path',argv)
	#--- write
	confParser.write(open('.env','w'))	
#	sfile = open('.env','w')
#	print('%s\n%s'%(argv,argv2nd),file=sfile)
#	sfile.close()
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
	jobname  = {
				0:'NiCoCrNatom100KTakeOneOutRlxd',
				1:'NiCoCrNatom200KTemp600Annealed', 
				2:'NiCoCrNatom100KTemp600',
				3:'NiCoCrNatom100KTemp600/dislocated/load850',
				}[3]
	DeleteExistingFolder = True
	readPath = os.getcwd() + {
								0:'/../testRuns/glassCo5Cr2Fe40Mn27Ni26',
								1:'/../lammpsRuns/AmirData/shengAnnealed/Temp600/dislocated',
							}[1] # --- source
	EXEC_DIR = '.'     #--- path for executable file
	durtn = '11:59:59'
	resources = {'mem':'128gb', 'partition':['o12h','a12h','i12h'][2],'nodes':1,'ppn':1}
#	argv = '%s'%readPath #--- don't change! 
#	argv2nd = "indx=15\ntemperature=600\nload=500" 
	PYFILdic = { 
		0:'pressFluc2nd.ipynb',
		}
	keyno = 0
#---
#---
	PYFIL = PYFILdic[ keyno ] 
	#--- update argV
	#---
	if DeleteExistingFolder:
		os.system( 'rm -rf %s' % jobname ) # --- rm existing
	# --- loop for submitting multiple jobs
	for counter in nruns:
		print(' i = %s' % counter)
		writPath = os.getcwd() + '/%s/Run%s' % ( jobname, counter ) # --- curr. dir
		os.system( 'mkdir -p %s' % ( writPath ) ) # --- create folder
		os.system( 'cp .env LammpsPostProcess*.py OvitosCna.py utility*.py %s' % ( writPath ) ) #--- cp python module
		makeOAR( writPath, 1, 1, durtn, PYFIL, readPath+"/Run%s"%counter) # --- make oar script
		os.system( 'chmod +x oarScript.sh; mv oarScript.sh .env %s; cp %s/%s %s' % ( writPath, EXEC_DIR, PYFIL, writPath ) ) # --- create folder & mv oar scrip & cp executable
		os.system( 'qsub -q %s -l nodes=%s:ppn=%s -l walltime=%s -N %s.%s -o %s -e %s -d %s  %s/oarScript.sh'\
			%( resources['partition'], resources['nodes'], resources['ppn'], durtn, jobname, counter, writPath, writPath, writPath , writPath ) ) # --- runs oarScript.sh!
											 

