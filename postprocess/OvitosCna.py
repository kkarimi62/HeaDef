import os
import sys
import ovito
import ovito.modifiers as md
import numpy as np
import ovito.io as io 
from ovito.vis import Viewport, TachyonRenderer, RenderSettings
from ovito.data import CutoffNeighborFinder, NearestNeighborFinder
import math
import pdb
import json


def GetNpairs(data, finder):        
    Npairs = 0 #---
    for index in range(data.number_of_particles):
        for neigh in finder.find(index):
            Npairs += (index<neigh.index)
    return Npairs


def GetPairAttrs(data, neigh,iatom):
#    return list(map(lambda x:data.particle_properties.particle_identifier.array[x.index],neigh))
    return list(map(lambda x:(iatom,x.index,x.distance,x.delta[0],x.delta[1],x.delta[2], x.pbc_shift[0],x.pbc_shift[1],x.pbc_shift[2]),neigh))

start_frame = 0

#--- command-line args
InputFile = sys.argv[1] #--- input lammps file 
OutputFile = sys.argv[2] #--- output
nevery = int(sys.argv[3]) #--- process. frequency
AnalysisType = int(sys.argv[4]) #--- 0:CommonNeighborAnalysis, 1:g(r), 2:d2min, 3:voronoi analysis, 4 & 6: neighbor list, 5: dislocation analysis, 7: convert to dump, 8: displacements, 9: Periodic Image 10: nearest neighbor finder 11: Wigner-Seitz algorithm 
#print('AnalysisType=',AnalysisType)
if AnalysisType == 11: 
    RefFile = sys.argv[5]
if AnalysisType == 8: 
    RefFile = sys.argv[5]
    use_frame_offset = False
    try:
        use_frame_offset = eval(sys.argv[6])
    except:
        pass
    if use_frame_offset:
       start_frame = 1
if AnalysisType == 3: #--- voronoi analysis
    radii=list(map(float,sys.argv[5:]))
if AnalysisType == 4 or AnalysisType == 6: #--- neighbor lists
    cutoff = float(sys.argv[5])
    if AnalysisType == 4:
#        natoms = int(sys.argv[6])
        natoms = np.loadtxt(sys.argv[6],dtype=int)
#        print('str=',natoms)
    elif AnalysisType == 6:
        atom_indices = np.array(list(map(int,sys.argv[6:])))
#        print(atom_indices)
if AnalysisType == 7:
    OutputFile_headers = sys.argv[5] 
if AnalysisType == 5:
    pbc_false = int(sys.argv[5]) 
if AnalysisType == 10:
    with open(sys.argv[5],'r') as fp:
        dataa =json.load(fp)
verbose = False

if verbose:
    print('InputFile=',InputFile)
# Load input data and create a data pipeline.
if AnalysisType == 7 or AnalysisType == 11:
    pipeline = io.import_file('%s'%(InputFile), multiple_frames = True, 
                             columns = ["Particle Type", "Position.X", "Position.Y", "Position.Z","Particle Identifier"])
else:
    pipeline = io.import_file('%s'%(InputFile), multiple_frames = True)

#pdb.set_trace()	
if verbose:
    print('num_frames=',pipeline.source.num_frames)

# Calculate per-particle displacements with respect to initial simulation frame
if AnalysisType == 0:
    cna = md.CommonNeighborAnalysisModifier()
    pipeline.modifiers.append(cna)

if AnalysisType == 9:
    pim = md.ShowPeriodicImagesModifier(adjust_box=True,unique_ids=True,replicate_x=True,replicate_y=True,replicate_z=True)
    pipeline.modifiers.append(pim)

#apply modifier
if AnalysisType == 1:
    cnm = md.CoordinationNumberModifier(cutoff = 10.0, number_of_bins = 200)
    pipeline.modifiers.append(cnm)
    sfile = open(OutputFile,'a')

if AnalysisType == 2:
    d2min = md.AtomicStrainModifier(
#                                    use_frame_offset=False,
                                    output_nonaffine_squared_displacements=True,
                                    eliminate_cell_deformation=True,
                                   )
    d2min.reference.load(InputFile)
    pipeline.modifiers.append(d2min)

if AnalysisType == 8:
    disp = md.CalculateDisplacementsModifier( use_frame_offset = use_frame_offset,

                                   )
    disp.reference.load(RefFile, multiple_frames = True)
    pipeline.modifiers.append(disp)

if AnalysisType == 3:
    # Set atomic radii (required for polydisperse Voronoi tessellation).
    #atypes = pipeline.source.particle_properties.particle_type.type_list
    type_property = pipeline.source.particle_properties.particle_type
#     print(radii)
#    for t in type_property.type_list:
#         print(t.id)
#        t.radius = radii[t.id-1]
    # Set up the Voronoi analysis modifier.
    voro = md.VoronoiAnalysisModifier(
                                    compute_indices = True,
                                    use_radii = False, #True,
                                    edge_count = 9, # Length after which Voronoi index vectors are truncated
                                    edge_threshold = 0.1
                                    )
    pipeline.modifiers.append(voro)

#--- neighbor list
if AnalysisType == 4 or AnalysisType == 6 or AnalysisType == 10:
    sfile = open(OutputFile,'ab')

if AnalysisType == 5:
#    ovito.data.SimulationCell(pbc=(False,False,False))
    disl = md.DislocationAnalysisModifier(line_coarsening_enabled=False,
                                         line_smoothing_enabled=False,
                                         )
    disl.input_crystal_structure = md.DislocationAnalysisModifier.Lattice.FCC
    pipeline.modifiers.append(disl)
    if pbc_false:
        print('pbc_false')
        pipeline.source.cell.pbc=(False, False, False)

if AnalysisType == 11:
    wsModifier = md.WignerSeitzAnalysisModifier(eliminate_cell_deformation=True,
                                         )
    wsModifier.reference.load(RefFile)
    pipeline.modifiers.append(wsModifier)

for frame, counter in zip(range(start_frame,pipeline.source.num_frames,nevery),range(pipeline.source.num_frames)):
    # This loads the input data for the current frame and
    # evaluates the applied modifiers:
    if verbose:
        print('frame=%s'%frame)
#    pipeline.compute(frame)
    data = pipeline.compute(frame)
    try:
        itime = pipeline.source.attributes['Timestep']
    except:
        pass
#    print(itime)
    if AnalysisType == 1:
        sfile.write('#ITIME\n%s\n'%itime)
        np.savetxt(sfile, cnm.rdf, header='r\tg(r)')




    #--- compute nearest neighbor
    if AnalysisType == 10:
        finder = NearestNeighborFinder(1, data)
        xyz_coords = dataa['query'][counter] #--- only if nevry==1
        nquery = int(len(xyz_coords)/3)
        xyz_coords = np.array(xyz_coords).reshape((nquery,3))
#        print(xyz_coords.shape)
#        pdb.set_trace()
#        print(dir(finder.find_at(xyz_coords[0]))
        if counter == 0:
            nearestAtoms = []
        nearestAtoms += [list(map(lambda x:[item.index for item in finder.find_at(x)],xyz_coords))]
#        for xyz in xyz_coords: 
#            for neigh in finder.find_at(xyz):
#            print(neigh.index, neigh.distance, neigh.delta)
#            sfile.write(b'ITIME: TIMESTEP\n%d\n'%itime)


    #--- compute neighbor list
    if AnalysisType == 4 or AnalysisType == 6:
        type_property = pipeline.source.particle_properties.particle_type
        finder = CutoffNeighborFinder(cutoff, data)
        if AnalysisType == 4:
            atom_indices = natoms #range(natoms)
#        elif AnalysisType == 6:
#            atom_indices = np.arange(data.number_of_particles)[filtr_atoms]
        neighList = list(map(lambda x: finder.find(x) , atom_indices )) #range(data.number_of_particles) ))
        zipp = zip(neighList,atom_indices) #data.number_of_particles))
        pairij = np.concatenate(list(map(lambda x: GetPairAttrs( data, x[0],x[1] ), zipp))) #,dtype=object)
        #
        indexi = list(map(int,pairij[:,0]))
        indexj = list(map(int,pairij[:,1]))
        atomi_id=data.particle_properties.particle_identifier.array[indexi]
        atomj_id=data.particle_properties.particle_identifier.array[indexj]
        atomi_type = type_property.array[indexi]
        atomj_type = type_property.array[indexj]
        #
        sfile.write(b'ITIME: TIMESTEP\n%d\n'%itime)
        sfile.write(b'ITEM: NUMBER OF ATOMS\n%d\n'%(len(indexi)))
        sfile.write(b'ITEM: BOX BOUNDS xy xz yz pp pp pp\n0.0\t0.0\t0.0\n0.0\t0.0\t0.0\n0.0\t0.0\t0.0\n')
        sfile.write(b'ITEM: ATOMS id\ttype\tJ\tJtype\tDIST\tDX\tDY\tDZ\tPBC_SHIFT_X\tPBC_SHIFT_Y\tPBC_SHIFT_Z\n')
#        pdb.set_trace()
        np.savetxt(sfile,np.c_[ atomi_id, atomi_type, atomj_id, atomj_type, pairij[:,2:]],
                   fmt='%i %i %i %i %7.6e %7.6e %7.6e %7.6e %i %i %i' )

#         for index in range(data.number_of_particles):
#             atomi_id = data.particle_properties.particle_identifier.array[index]
#             atomi_type = type_property.array[index]
#             if atomi_id == 1 or atomi_id == 2: print("Neighbors of particle %i:" % atomi_id)
# #            pdb.set_trace()
            # Iterate over the neighbors of the current particle:
#             for neigh in finder.find(index):
#                 atomj_id = data.particle_properties.particle_identifier.array[neigh.index]
#                 atomj_type = type_property.array[neigh.index]
#                 if atomi_id == 1 or atomi_id == 2: print("%i %i:" %(atomi_id,atomj_id))
#                 if atomi_id < atomj_id:
# #                print(neigh.index, neigh.distance, neigh.delta, neigh.pbc_shift)
#                     sfile.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(atomi_id,atomi_type,atomj_id, atomj_type, neigh.distance, neigh.delta[0],neigh.delta[1],neigh.delta[2], neigh.pbc_shift[0],neigh.pbc_shift[1],neigh.pbc_shift[2]))

    # Access computed Voronoi indices as NumPy array.
    # This is an (N)x(edge_count) array.
#     if AnalysisType == 3:
#         voro_indices = pipeline.output.particle_properties['Voronoi Index'].array

#    pdb.set_trace()
if AnalysisType == 10:
    with open(OutputFile,'w') as fp:
        #--- output as json
        dictionary={'frameIndex':list(range(len(nearestAtoms))),'nearestAtoms':nearestAtoms}
        json.dump(dictionary, fp)


if AnalysisType == 1 or AnalysisType == 4:
    sfile.close()

#--- export data
if AnalysisType == 0:
    io.export_file( pipeline, OutputFile, "lammps_dump",\
                    columns = ["Particle Identifier", "Particle Type", "Position.X","Position.Y","Position.Z",\
                               "Structure Type"],
                     start_frame = 0,
#                     end_frame = pipeline.source.num_frames,
                     every_nth_frame = nevery,
                     multiple_frames=True )
if AnalysisType == 2:
    io.export_file( pipeline, OutputFile, "lammps_dump",\
                    columns = ["Particle Identifier", "Particle Type", "Position.X","Position.Y","Position.Z",\
                               "Nonaffine Squared Displacement"],
                     start_frame = 0,
#                     end_frame = pipeline.source.num_frames,
                     every_nth_frame = nevery,
                     multiple_frames=True )

if AnalysisType == 8:
    if not use_frame_offset:
        io.export_file( pipeline, OutputFile, "lammps_dump",\
                    columns = ["Particle Identifier", "Particle Type", "Position.X","Position.Y","Position.Z",\
                               "Displacement.X","Displacement.Y","Displacement.Z"],
                     start_frame = start_frame,
#                     end_frame = pipeline.source.num_frames,
                     every_nth_frame = nevery,
                     multiple_frames=True )

    else:
        xstr = ''
        for frame, counter in zip(range(start_frame,pipeline.source.num_frames,nevery),range(pipeline.source.num_frames)):
            io.export_file( pipeline, '%s*'%OutputFile, "lammps_dump",\
                    columns = ["Particle Identifier", "Particle Type", "Position.X","Position.Y","Position.Z",\
                               "Displacement.X","Displacement.Y","Displacement.Z"],
                     frame=frame )
            xstr += ' %s%s'%(OutputFile,frame)
        os.system('cat %s > %s'%(xstr,OutputFile))
        os.system('rm %s'%xstr)

if AnalysisType == 3: 
    io.export_file( pipeline, OutputFile, "lammps_dump",\
                    columns = ["Particle Identifier", "Particle Type", "Position.X","Position.Y","Position.Z",\
                               "Voronoi Index.0","Voronoi Index.1","Voronoi Index.2",\
                               "Voronoi Index.3","Voronoi Index.4","Voronoi Index.5",\
                               "Voronoi Index.6","Voronoi Index.7","Voronoi Index.8", "Atomic Volume"],
                     start_frame = 0,
#                     end_frame = pipeline.source.num_frames,
                     every_nth_frame = nevery,

                    multiple_frames=True 
                  )   

if AnalysisType == 5: 
    # data.particle_properties['Cluster'].array
#    pdb.set_trace()
#    print('OutputFile=',OutputFile)
    io.export_file( pipeline, '%s.*'%OutputFile, "ca",
                     start_frame = 0,
#                     end_frame = pipeline.source.num_frames,
                     every_nth_frame = nevery,
                    multiple_frames=True 
                  )   
    io.export_file( pipeline, '%s.xyz'%OutputFile, "lammps_dump",
                    columns = list(data.particle_properties.keys())+["Position.X","Position.Y","Position.Z"], #["Particle Identifier", "Particle Type", "Position.X","Position.Y","Position.Z","Structure Type","Cluster"],
                     start_frame = 0,
#                     end_frame = pipeline.source.num_frames,
                     every_nth_frame = nevery,
                    multiple_frames=True 
                  )   

if AnalysisType == 9:
    io.export_file( pipeline, OutputFile, "lammps_dump",\
                    columns = ["Particle Identifier", "Particle Type", "Position.X","Position.Y","Position.Z"],
                     start_frame = 0,
#                     end_frame = pipeline.source.num_frames-1,
                     every_nth_frame = nevery,
                    multiple_frames=True 
                  )   
if AnalysisType == 7: 
    io.export_file( pipeline, OutputFile, "lammps_dump",\
                    columns = ["Particle Identifier", "Particle Type", "Position.X","Position.Y","Position.Z"],
                     start_frame = 0,
#                     end_frame = pipeline.source.num_frames-1,
                     every_nth_frame = nevery,
                    multiple_frames=True 
                  )   
    io.export_file(pipeline, OutputFile_headers, "txt", multiple_frames=True,
#         columns = ["Frame", "SelectExpression.num_selected"])
        columns=list(pipeline.source.attributes.keys()
    ))
# Export the computed RDF data to a text file.

if AnalysisType == 11: 
    io.export_file( pipeline, OutputFile, "lammps_dump",\
                    columns = ["Particle Identifier", "Particle Type", "Position.X","Position.Y","Position.Z","Occupancy"],
                     start_frame = 0,
#                     end_frame = pipeline.source.num_frames-1,
                     every_nth_frame = nevery,
                    multiple_frames=True 
                  )   

'''
pipeline.dataset.anim.frames_per_second = 60
pipeline.add_to_scene()
vp = Viewport()

vp.type = Viewport.Type.PERSPECTIVE

#vp.camera_pos = (735.866,-725.04,1001.35)
vp.camera_pos = (118.188,-157.588,131.323)

#vp.camera_dir = (-0.49923, 0.66564, -0.5547)
vp.camera_dir = (-0.49923,0.66564,-0.5547) 

vp.fov = math.radians(35.0)

tachyon = TachyonRenderer() #shadows=False, direct_light_intensity=1.1)

rs = RenderSettings(size=(600,600), filename="image.mov",
#                   custom_range=(0,100),
                    everyNthFrame=1,
                    range = RenderSettings.Range.ANIMATION, #CUSTOM_INTERVAL, #RenderSettings.Range.ANIMATION,  
                    renderer=tachyon,
                    )

vp.render(rs)
'''
