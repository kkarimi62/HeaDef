import numpy as np
import pandas as pd
import pdb
import sys
import os


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
            dff.iloc[indxxx[itype]]['type'] = ntype - itype
            indices = list(set(indices)-set(indxxx[itype]))
            sizeTot -= size		
        atoms = lp.Atoms( **dff.to_dict(orient='series') )
	
    #--- write data file
    lp.WriteDataFile(atoms,box,mass).Write(LmpInput)

pathlib = sys.argv[1]
sys.path.append(pathlib)
import LammpsPostProcess as lp
 
if 1: #Atomsk:
    mass={1:58.693, # Ni
        2:58.933195, # Co
        3:51.9961 #Cr,
       } 
	#
    a = float(sys.argv[2]) #3.52
    lx, ly, lz = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]) #40.0, 20.0, 40.0 Angstrom
#   m,n,k = 90, 30, 4
    m,n,k = int(lx/a*4.0**(1.0/3)), int(ly/a), int(lz/a)
    #
    bmag = a / 2.0 ** 0.5
    os.system('rm *.cfg *.lmp *.xyz')
    os.system('atomsk --create fcc %s Ni orient 110 -111 1-12 Al_unitcell.cfg'%(a))
#    os.system('atomsk Al_unitcell.cfg -duplicate %s %s %s Al_supercell.cfg'%(m,n,k))
    os.system('atomsk Al_unitcell.cfg -duplicate %s %s %s -deform X 0.0125 0.0 bottom.xsf'%(m,int(n/2),k))
    os.system('atomsk Al_unitcell.cfg -duplicate %s %s %s -deform X -0.012195122 0.0 top.xsf'%(m+1,int(n/2),k))
    os.system('atomsk --merge Y 2 bottom.xsf top.xsf data.cfg')
#    os.system('atomsk Al_supercell.cfg -dislocation 0.51*box 0.51*box edge_rm Z Y %s 0.33 data.cfg'%(bmag))
    os.system('atomsk data.cfg -center com final.cfg')
    os.system('atomsk final.cfg lmp')
    #
    WriteDataFile('final.lmp',mass, sys.argv[6])

