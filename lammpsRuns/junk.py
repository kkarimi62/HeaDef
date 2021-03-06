def makeOAR( EXEC_DIR, node, core, time ):
	someFile = open( 'oarScript.sh', 'w' )
	print >> someFile, '#!/bin/bash\n'
	print >> someFile, 'EXEC_DIR=%s\n' %( EXEC_DIR )
	print >> someFile, 'MEAM_library_DIR=%s\n' %( MEAM_library_DIR )
	print >> someFile, 'module load mpich/3.2.1-gnu\n'

	#--- run python script 
#	 print >> someFile, "$EXEC_DIR/%s < in.txt -var OUT_PATH %s -var MEAM_library_DIR %s"%( EXEC, OUT_PATH, MEAM_library_DIR )
#	cutoff = 1.0 / rho ** (1.0/3.0)
	for script,var,indx, execc in zip(Pipeline,Variables,range(100),EXEC):
		if execc == 'lmp': #_mpi' or EXEC == 'lmp_serial':
			print >> someFile, "mpirun -np %s $EXEC_DIR/lmp_mpi < %s -echo screen -var OUT_PATH %s -var PathEam %s -var INC %s %s"%(nThreads*nNode, script, OUT_PATH, MEAM_library_DIR, SCRPT_DIR, var)
		elif execc == 'py':
			print >> someFile, "python3 %s %s"%(script, var)
			
	someFile.close()										  


if __name__ == '__main__':
	import os
	import numpy as np

	nruns	 = 1
	#
	nThreads = [1,9][1]
	nNode	 = 1
	#
	jobname  = {
				1:'NiNatom100KReplaceCoRlxd',
				2:'NiCoCrNatom100KTemp300', 
				3:'NiNatom1KT0EdgeDisl', 
				4:'NiCoCrNatom1000KEdgeDisl', 
				5:'NiCoCrNatom200KTemp600Annealed', 
				6:'NiCoCrNatom100KTemp300Gdot4',
				7:'NiNatom1KT0EdgeDisl2nd',
				8:'NiCoCrNatom100KT0Elastic',
				9:'NiCoCrNatom100KAnnealedT600Elastic',
			   }[9]
	sourcePath = os.getcwd() +\
				{	
					1:'/../postprocess/NiCoCrNatom1K',
					2:'/NiCoCrNatom100K',
					3:'/NiCoCrNatom100KTemp300',
					4:'/junk',
					5:'/../postprocess/NiNatom1KEdgeDisl',
					6:'/../postprocess/NiCoCrNatom1K', 
					7:'/../postprocess/NiCoCrNatom1000K', 
					8:'/NiCoCrNatom200KTemp600', 
					9:'/NiNatom1KT0EdgeDisl',
					10:'/NiCoCrNatom10KT0Elastic',
				}[4] #--- must be different than sourcePath
        #
	sourceFiles = { 0:False,
					1:['Equilibrated_300.dat'],
					2:['data.txt','ScriptGroup.txt'],
					3:['data.txt'], 
					4:['data_minimized.txt'],
					5:['data_init.txt','ScriptGroup.0.txt'], #--- only one partition! for multiple ones, use 'submit.py'
				 }[0] #--- to be copied from the above directory
	#
	EXEC_DIR = '/home/kamran.karimi1/Project/git/lammps2nd/lammps/src' #--- path for executable file
	#
	MEAM_library_DIR='/home/kamran.karimi1/Project/git/lammps2nd/lammps/potentials'
	#
	SCRPT_DIR = os.getcwd()+'/lmpScripts/'+{1:'Ni', 2:'NiCoCr'}[2]
	#
	SCRATCH = None
	OUT_PATH = '.'
	if SCRATCH:
		OUT_PATH = '/scratch/${SLURM_JOB_ID}'
	#--- py script must have a key of type str!
	LmpScript = {	0:'in.PrepTemp0',
				 	1:'relax.in', 
					2:'relaxWalls.in', 
					7:'in.Thermalization', 
					4:'in.vsgc', 
					5:'in.minimization', 
					6:'in.shearDispTemp', 
					8:'in.shearLoadTemp',
					9:'in.elastic',
					10:'in.elasticSoftWall',
					'p0':'partition.py',
					'p1':'WriteDump.py',
					'p2':'DislocateEdge.py',
				} 
	#
	Variable = {
				0:' -var natoms 100000 -var cutoff 3.52 -var ParseData 0  -var DumpFile DumpMin.xyz -var WriteData data_init.txt',
				6:' -var T 300 -var DataFile Equilibrated_300.dat',
				4:' -var T 600.0 -var t_sw 20.0 -var DataFile Equilibrated_600.dat -var nevery 1000 -var ParseData 1 -var WriteData swapped_600.dat', 
				5:' -var DataFile data.txt -var buff 3.0 -var DumpFile dumpMin.xyz -var nevery 1000 -var ParseData 1 -var WriteData data_minimized.txt', 
				7:' -var buff 0.0 -var T 600.0 -var teq 200.0 -var nevery 1000 -var ParseData 1 -var DataFile data_init.txt -var DumpFile dumpThermalized.xyz -var WriteData Equilibrated_600.dat',
				8:' -var buff 3.0 -var T 0.1 -var sigm 1.5 -var sigmdt 0.01 -var ParseData 1 -var DataFile Equilibrated_300.dat -var DumpFile dumpSheared.xyz',
				9:' -var natoms 1000 -var cutoff 3.52 -var ParseData 1',
				10:' -var ParseData 1 -var DataFile swapped_600.dat',
				'p0':' data_init.txt 10.0 %s'%(os.getcwd()+'/../postprocess'),
				'p1':' data_init.txt ElasticConst.txt DumpFileModu.xyz %s'%(os.getcwd()+'/../postprocess'),
				'p2':' %s 3.52 40.0 20.0 40.0 data.txt'%(os.getcwd()+'/../postprocess'),
				} 
	#--- different scripts in a pipeline
	indices = {
				0:['p2',5,7,8], #--- put disc. by atomsk, minimize, thermalize, and shear
				1:[9],     #--- elastic constants
				2:[0,10],	   #--- local elastic constants (zero temp)
				3:[0,7,4,10],	   #--- local elastic constants (annealed)
			  }[3]
	Pipeline = list(map(lambda x:LmpScript[x],indices))
	Variables = list(map(lambda x:Variable[x], indices))
	EXEC = list(map(lambda x:'lmp' if type(x) == type(0) else 'py', indices))	
	#
	EXEC_lmp = ['lmp_mpi','lmp_serial'][0]
	durtn = ['23:59:59','00:59:59'][0]
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
			os.system( 'ln -s %s/%s %s' % ( EXEC_DIR, EXEC_lmp, path ) ) # --- create folder & mv oar scrip & cp executable
		#---
		for script,indx in zip(Pipeline,range(100)):
#			os.system( 'cp %s/%s %s/lmpScript%s.txt' %( SCRPT_DIR, script, writPath, indx) ) #--- lammps script: periodic x, pxx, vy, load
			os.system( 'ln -s %s/%s %s' %( SCRPT_DIR, script, writPath) ) #--- lammps script: periodic x, pxx, vy, load
		if sourceFiles: 
			for sf in sourceFiles:
				os.system( 'ln -s %s/Run%s/%s %s' %(sourcePath, irun, sf, writPath) ) #--- lammps script: periodic x, pxx, vy, load
		#---
		makeOAR( path, 1, nThreads, durtn) # --- make oar script
		os.system( 'chmod +x oarScript.sh; mv oarScript.sh %s' % ( writPath) ) # --- create folder & mv oar scrip & cp executable
		os.system( 'sbatch --partition=%s --mem=%s --time=%s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
						    --chdir %s -c %s -n %s %s/oarScript.sh >> jobID.txt'\
						   % ( partition, mem, durtn, jobname, counter, jobname, counter, jobname, counter \
						       , writPath, nThreads, nNode, writPath ) ) # --- runs oarScript.sh! 
		counter += 1
											 
	os.system( 'mv jobID.txt %s' % ( os.getcwd() + '/%s' % ( jobname ) ) )
