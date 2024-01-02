import os
import sys
import numpy as np
#
import ovito
import ovito.modifiers as md
import ovito.io as io

def Loop( start_frame, nframes, nevery, pipeline,verbose):
#--- loop over frames
	for frame, counter in zip(range(start_frame,nframes,nevery),range(pipeline.source.num_frames)):
		if verbose:
			print('frame=%s'%frame)
		pipeline.compute(frame)
		try:
			itime = pipeline.source.attributes['Timestep']
		except:
			pass
	return frame

def PrintDisp(pipeline,OutputFile,start_frame,frame,nevery,use_frame_offset):
	#--- save displacements
	os.system('mkdir disp')
	if not use_frame_offset:
		io.export_file( pipeline, 'disp/%s'%OutputFile, "lammps_dump",\
					columns = ["Particle Identifier", "Particle Type", "Position.X","Position.Y","Position.Z",\
							   "Displacement.X","Displacement.Y","Displacement.Z"],
					 start_frame = start_frame,
					 end_frame = frame,
					 every_nth_frame = nevery,
					 multiple_frames=True )

	else:
		xstr = ''
		for frame, counter in zip(range(start_frame,nframes,nevery),range(pipeline.source.num_frames)):
			io.export_file( pipeline, 'disp/%s*'%OutputFile, "lammps_dump",\
					columns = ["Particle Identifier", "Particle Type", "Position.X","Position.Y","Position.Z",\
							   "Displacement.X","Displacement.Y","Displacement.Z"],
					 frame=frame )
			xstr += ' disp/%s%s'%(OutputFile,frame)
		os.system('cat %s > disp/%s'%(xstr,OutputFile))
		os.system('rm %s'%xstr)

def main():

	kwargs             = dict(zip(sys.argv[1:][1::3],sys.argv[1:][2::3]))
	InputFile          = kwargs['InputFile'] 
	RefFileDisp        = kwargs['RefFileDisp']
	RefFileDefect      = kwargs['RefFileDefect']
	OutputFile         = kwargs['OutputFile'] 
	OutputFile_headers = kwargs['OutputFile_headers'] 
	WignerSeitz        = eval( kwargs['WignerSeitz'] ) 
	nevery             = int(kwargs['nevery']) 
	verbose            = eval(kwargs['verbose']) if 'verbose' in kwargs else False
	use_frame_offset   = eval(kwargs['use_frame_offset'])
 
   #--- parse data file
	pipeline           = io.import_file('%s'%(InputFile), multiple_frames = True, 
                             columns = ["Particle Type", "Position.X", "Position.Y", "Position.Z","Particle Identifier"])

	#--- no. of frames
	start_frame = 0
	if 'nframes' in kwargs:
		nframes = np.min([int(kwargs['nframes']),pipeline.source.num_frames])
	else:
		nframes = pipeline.source.num_frames
	if verbose:
		print('nframes=',nframes)

	#--- displacements
	if verbose:
		print('compute displacements ...')
	disp = md.CalculateDisplacementsModifier( use_frame_offset = use_frame_offset )
	disp.reference.load( RefFileDisp, multiple_frames = True)
	pipeline.modifiers.append( disp )
	last_frame = Loop( start_frame, nframes, nevery, pipeline, verbose)
	PrintDisp(pipeline,OutputFile,start_frame,last_frame,nevery,use_frame_offset)
	if verbose:
		print('output in folder disp')

	#--- print headers
	io.export_file(pipeline, 'disp/%s'%OutputFile_headers, "txt", multiple_frames=True,
                   start_frame = 0, end_frame = last_frame,
                   columns=list(pipeline.source.attributes.keys()))

	#--- wrap coordinates
#	wrModifier = md.WrapPeriodicImagesModifier()
#	pipeline.modifiers.append(wrModifier)
#	last_frame = Loop( start_frame, nframes, nevery, pipeline, verbose)

	#--- defect analysis
	if WignerSeitz:
		if verbose:
			print('Wigner-Seitz Analysis ...')
		wsModifier = md.WignerSeitzAnalysisModifier(eliminate_cell_deformation=True)
		wsModifier.reference.load( RefFileDefect )
		pipeline.modifiers.append(wsModifier)
		last_frame = Loop( start_frame, nframes, nevery, pipeline, verbose)
		#--- save occupancy 
		os.system('mkdir occupancy')
		io.export_file( pipeline, 'occupancy/%s'%OutputFile, "lammps_dump",
				   columns = ["Particle Identifier", "Particle Type", "Position.X","Position.Y","Position.Z","Occupancy"],
				   start_frame = 0,
				  end_frame = last_frame, 
				 every_nth_frame = nevery,
			   multiple_frames=True 
			 )   
		if verbose:
			print('output in folder occupancy')

if __name__ == '__main__':
	main()
