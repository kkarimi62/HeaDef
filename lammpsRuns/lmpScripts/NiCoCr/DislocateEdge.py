import numpy as np
import pandas as pd
import pdb
import sys
import os
import pdb

def WriteDataFile(AtomskOutpt, mass, LmpInput):
    #--- read data file
    lmpData = lp.ReadDumpFile( AtomskOutpt )
    lmpData.ReadData()
    #--- atom obj
    atoms = lp.Atoms( **lmpData.coord_atoms_broken[0].to_dict(orient='series') )
    #--- box
    box = lp.Box( BoxBounds = lmpData.BoxBounds[0],AddMissing = np.array([0.0,0.0,0.0] ) )
    #--- wrap
    wrap = lp.Wrap( atoms, box )
    wrap.WrapCoord()
    wrap.Set( atoms )
    #--- center
    #--- add box bounds
    rcent = np.matmul(box.CellVector,np.array([.5,.5,.5]))
    box.CellOrigin -= rcent
    loo=box.CellOrigin
    hii=box.CellOrigin+np.matmul(box.CellVector,np.array([1,1,1]))
    box.BoxBounds=np.c_[loo,hii,np.array([0,0,0])]

    atoms.x -= rcent[0]
    atoms.y -= rcent[1]
    atoms.z -= rcent[2]

    if len(mass) > 1: #--- multi-component alloy: assign random types
        dff=pd.DataFrame(atoms.__dict__)
        dff['type']=1
        indices = dff.index
        ntype=len(mass)
        sizeTot = len(dff)
        size = int(np.floor((1.0*sizeTot/ntype)))
        assert size * ntype <= sizeTot
        indxxx = {}
        for itype in range(ntype-1):
            indxxx[itype] = np.random.choice(indices, size=size, replace=None)
#            dff.iloc[indxxx[itype]]['type'] = ntype - itype
            row_indexer = indxxx[itype]
            col_indexer = 'type'
            dff.loc[row_indexer,col_indexer] = ntype - itype 
            indices = list(set(indices)-set(indxxx[itype]))
            sizeTot -= size		
        atoms = lp.Atoms( **dff.to_dict(orient='series') )
#        pdb.set_trace()	
    #--- write data file
    lp.WriteDataFile(atoms,box,mass).Write(LmpInput)

pathlib = sys.argv[1]
sys.path.append(pathlib)
import LammpsPostProcess as lp
 
if 1: #Atomsk:
    mass={1:58.693, # Ni
#        2:58.933195, # Co
#        3:51.9961 #Cr,
       } 
	#
    a = float(sys.argv[2]) #3.52
    lx, ly, lz = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]) #40.0, 20.0, 40.0 Angstrom
    method = sys.argv[7]
    m,n,k = int(lx/a*4.0**(1.0/3)), int(ly/a), int(lz/a)
    #
    var1=int(n/2)
    var2=m+1
    #
    bmag = a / 2.0 ** 0.5
    os.system('rm *.cfg *.lmp *.xyz *.xsf')
    #--- Crystallographic orientation of the system
    os.system('atomsk --create fcc %s Ni orient 110 -111 1-12 Al_unitcell.cfg'%a)
    os.system('atomsk Al_unitcell.cfg -duplicate %s %s %s Al_supercell.cfg'%(m,n,k))
    #--- Introduce an edge dislocation at constant number of atoms
    if method == '2':
        os.system('atomsk Al_supercell.cfg -dislocation 0.51*box 0.51*box edge Z Y %s 0.33 data.cfg'%(bmag))
    #--- Insert a half-plane above the glide plane
    if method == '3':
        os.system('atomsk Al_supercell.cfg -dislocation 0.51*box 0.51*box edge_add Z Y %s 0.33 data.cfg'%(bmag))
    #---  Remove a half-plane below the glide plane
    if method == '4':
        os.system('atomsk Al_supercell.cfg -dislocation 0.51*box 0.51*box edge_rm Z Y %s 0.33 data.cfg'%(bmag))
    #--- Construct a dislocation by superimposing two crystals
    if method == '5':
        epsilon = 0.5 / m
        os.system('atomsk Al_unitcell.cfg -duplicate %s %s %s -deform X %s 0.0 bottom.xsf'%(m,var1,k,epsilon))
        epsilon = -0.5 / (m+1)
        os.system('atomsk Al_unitcell.cfg -duplicate %s %s %s -deform X %s 0.0 top.xsf'%(var2,var1,k,epsilon))
        os.system('atomsk --merge Y 2 bottom.xsf top.xsf data.cfg') 

    os.system('atomsk data.cfg -center com final.cfg')
    os.system('atomsk final.cfg lmp')
    #
    WriteDataFile('final.lmp',mass, sys.argv[6])
