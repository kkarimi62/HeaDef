from backports import configparser

def makeOAR( EXEC_DIR, node, core, tpartitionime, PYFIL, argv):
	#--- parse conf. file
	confParser = configparser.ConfigParser()
	confParser.read('configuration.ini')
	#--- set parameters
	confParser.set('parameters','temperature','800')
	confParser.set('parameters','load','450')
	confParser.set('input files','path',argv.split()[1])
	confParser.set('py library path','py_lib',argv.split()[0])
	#--- write
	confParser.write(open('configuration.ini','w'))	
	#--- set environment variables

	someFile = open( 'oarScript.sh', 'w' )
	print('#!/bin/bash\n',file=someFile)
	print('EXEC_DIR=%s\n'%( EXEC_DIR ),file=someFile)
	print('module load python/anaconda3-2018.12\nsource /global/software/anaconda/anaconda3-2018.12/etc/profile.d/conda.sh\nconda activate gnnEnv2nd ',file=someFile)
#	print >> someFile, 'papermill --prepare-only %s/%s ./output.ipynb %s %s'%(EXEC_DIR,PYFIL,argv,argv2nd) #--- write notebook with a list of passed params
	if convert_to_py:
		print('ipython3 py_script.py\n',file=someFile)
	else:	 
		print('jupyter nbconvert --execute $EXEC_DIR/%s --to html --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html'%(PYFIL), file=someFile)
	someFile.close()										  
#
if __name__ == '__main__':
	import os
#
	nruns	 = range(1)
	jobname  = {
					1:'NiCoCrNatom100KTemp800RhoFluc',
					2:'NiCoCrNatom100KTemp800RhoFlucRss', 
					3:'NiNatom100KTakeOneOut',
					4:'NiNatom100KReplaceCr',
					5:'NiNatom100KReplaceCo',
					6:'NiNatom1K',
					7:'NiCoCrNatom1000K', 
					8:'NiNatom1KEdgeDisl',
					9:'NiCoCrNatom1KT0Elastic',
					10:'FeNiT300Elasticity',
					11:'NiCoCrNatom100KTemp800sroFarkas',
					12:'indentation2nd',
					13:'compression',
					14:'tension2nd',
					15:'Annealed_before_indentation',
					16:'anneled_before_compression',
					17:'RSS_before_indentation',
					18:'RSS_compressed',
					19:'nicocrNatom100KMultipleTempIrradiatedAnneal/benchmark/temp0',
					20:'AmirData/NiAl/Swapped_300',
				}[20]
	DeleteExistingFolder = True
	readPath = os.getcwd() + {
								1:'/../lammpsRuns/AmirData/shengAnnealed/Temp800', #--- source
								2:'/../lammpsRuns/AmirData/farkas', #--- source
								3:'/../lammpsRuns/AmirData/indentation', #--- source
								4:'/../lammpsRuns/AmirData/compression', #--- source
								5:'/../lammpsRuns/AmirData/tension', 
								15:'/../lammpsRuns/AmirData/Annealed_before_indentation',
								16:'/../lammpsRuns/AmirData/anneled_before_compression',
								17:'/../lammpsRuns/AmirData/RSS_before_indentation',
								18:'/../lammpsRuns/AmirData/RSS_compressed',
								19:'/../lammpsRuns/nicocrNatom100KMultipleTempIrradiatedAnneal/benchmark/temp0',
								20:'/../lammpsRuns/AmirData/NiAl/Swapped_300',
							}[20]
	EXEC_DIR = '.'     #--- path for executable file
	home_directory = os.path.expanduser( '~' )
	py_library_directory = '%s/Project/git/HeaDef/postprocess'%home_directory 
	durtn = '23:59:59'
	mem = '512gb'
	partition = ['cpu2019','bigmem','parallel','single'][1]
	argv = "%s %s"%(py_library_directory,readPath) #--- don't change! 
	PYFILdic = { 
		0:'pressFluc.ipynb',
		1:'partition.ipynb',
		2:'Moduli.ipynb',
		3:'pressFluc2nd.ipynb',
		}
	keyno = 3
	convert_to_py = True
#---
#---
	PYFIL = PYFILdic[ keyno ] 
	#--- update argV
	if convert_to_py:
		os.system('jupyter nbconvert --to script %s --output py_script\n'%PYFIL)
		PYFIL = 'py_script.py'
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
		os.system( 'cp configuration.ini LammpsPostProcess*.py OvitosCna.py utility*.py %s' % ( writPath ) ) #--- cp python module
		makeOAR( writPath, 1, 1, durtn, PYFIL, argv+"/Run%s"%init) # --- make oar script
		os.system( 'chmod +x oarScript.sh; cp oarScript.sh configuration.ini %s; cp %s/%s %s' % ( writPath, EXEC_DIR, PYFIL, writPath ) ) # --- create folder & mv oar scrip & cp executable
		jobname0 = jobname.split('/')[0]
		os.system( 'sbatch --partition=%s --mem=%s --time=%s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
						    --chdir %s -c %s -n %s %s/oarScript.sh'\
						   % ( partition, mem, durtn, jobname0, counter, jobname0, counter, jobname0, counter \
						       , writPath, 1, 1, writPath ) ) # --- runs oarScript.sh!
											 

