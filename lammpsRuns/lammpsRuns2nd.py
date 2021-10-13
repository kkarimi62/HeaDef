def makeOAR( EXEC_DIR, node, core, time ):
	someFile = open( 'oarScript.sh', 'w' )
	print >> someFile, '#!/bin/bash\n'
	print >> someFile, 'EXEC_DIR=%s\n' %( EXEC_DIR )
	print >> someFile, 'MEAM_library_DIR=%s\n' %( MEAM_library_DIR )
	print >> someFile, 'module load mpich/3.2.1-gnu\n'

	#--- run python script 
	OUT_PATH = '.'
	if SCRATCH:
		OUT_PATH = '/scratch/${SLURM_JOB_ID}'
#	 print >> someFile, "$EXEC_DIR/%s < in.txt -var OUT_PATH %s -var MEAM_library_DIR %s"%( EXEC, OUT_PATH, MEAM_library_DIR )
#	cutoff = 1.0 / rho ** (1.0/3.0)
	if EXEC == 'lmp_mpi' or EXEC == 'lmp_serial':
		print >> someFile, "mpirun -np %s $EXEC_DIR/%s < %s -echo screen -var OUT_PATH %s -var PathEam %s %s"%(nThreads*nNode, EXEC, 'lmpScript.txt', OUT_PATH, MEAM_library_DIR, Variables)
	someFile.close()										  


if __name__ == '__main__':
	import os
        import numpy as np

	nruns	 = 1
	#
	nThreads = 9
	nNode	 = 1
	#
	jobname  = ['NiNatom100KReplaceCoRlxd','NiCoCrNatom100KTemp300', 'NiCoCrNatom100KTemp300Annealed', 'NiCoCrT0EdgeDisl', 'NiNatom1KEdgeDisl','NiCoCrNatom1KEdgeDisl', 'NiCoCrNatom1000KEdgeDisl', 'NiCoCrNatom400KTemp600Annealed' ][-1]
	sourcePath = os.getcwd() +\
				['/../postprocess/NiCoCrNatom1K','/NiCoCrNatom100K','/NiCoCrNatom100KTemp300','/junk','/../postprocess/NiNatom1K','/../postprocess/NiCoCrNatom1K', '/../postprocess/NiCoCrNatom1000K', '/NiCoCrNatom400KTemp600'][-1] #--- must be different than sourcePath
        #
        sourceFiles = [['Equilibrated_600.dat'],['data.txt','ScriptGroup.txt'],['data.txt']][0] #--- to be copied from the above directory
	#
	EXEC_DIR = '/home/kamran.karimi1/Project/git/lammps2nd/lammps/src' #--- path for executable file
	#
	MEAM_library_DIR='/home/kamran.karimi1/Project/git/lammps2nd/lammps/potentials'
	SCRPT_DIR = os.getcwd()+'/lmpScripts'
	#
	LmpScript = ['Ni/relax.in', 'Ni/relaxWalls.in', 'NiCoCr/relax.in','NiCoCr/Thermalization.lmp', 'NiCoCr/vsgc.lmp', 'Ni/minimization_edge.lmp','NiCoCr/minimization_edge.lmp'][4] #--- [pbc, rigid walls,] 
	Variables = [' -var T 600 -var DataFile Equilibrated_600.dat',' -var buff 6.0 -var nevery 10'][0] 
	#
	EXEC = ['lmp_mpi','lmp_serial'][0]
	durtn = '167:59:59'
	SCRATCH = None
	mem = '8gb'
	partition = ['gpu-v100','parallel','cpu2019','single'][1]

	#---
	os.system( 'rm -rf %s' % jobname ) #--- rm existing
	os.system( 'rm jobID.txt' )
	# --- loop for submitting multiple jobs
	counter = 0
	for irun in xrange( nruns ):
#               cutoff = cutoffs[ irun ]
		print ' i = %s' % counter
		writPath = os.getcwd() + '/%s/Run%s' % ( jobname, counter ) # --- curr. dir
		os.system( 'mkdir -p %s' % ( writPath ) ) # --- create folder
		if irun == 0: #--- cp to directory
			path=os.getcwd() + '/%s' % ( jobname)
			os.system( 'cp %s/%s %s' % ( EXEC_DIR, EXEC, path ) ) # --- create folder & mv oar scrip & cp executable
		#---
		os.system( 'cp %s/%s %s/lmpScript.txt' %( SCRPT_DIR, LmpScript, writPath) ) #--- lammps script: periodic x, pxx, vy, load
                for sf in sourceFiles:
        		os.system( 'cp %s/Run%s/%s %s' %(sourcePath, irun, sf, writPath) ) #--- lammps script: periodic x, pxx, vy, load
		#---
		makeOAR( path, 1, nThreads, durtn) # --- make oar script
		os.system( 'chmod +x oarScript.sh; mv oarScript.sh %s' % ( writPath) ) # --- create folder & mv oar scrip & cp executable
		os.system( 'sbatch --partition=%s --mem=%s --time=%s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
						    --chdir %s -c %s -n %s %s/oarScript.sh >> jobID.txt'\
						   % ( partition, mem, durtn, jobname, counter, jobname, counter, jobname, counter \
						       , writPath, nThreads, nNode, writPath ) ) # --- runs oarScript.sh! 
		counter += 1
											 
	os.system( 'mv jobID.txt %s' % ( os.getcwd() + '/%s' % ( jobname ) ) )
