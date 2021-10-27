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

    #--- write data file
    lp.WriteDataFile(atoms,box,mass).Write(LmpInput)

pathlib = sys.argv[1]
sys.path.append(pathlib)
import LammpsPostProcess as lp
 
if 1: #Atomsk:
    a = float(sys.argv[2]) #3.52
    lx, ly, lz = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]) #40.0, 20.0, 40.0 Angstrom
#   m,n,k = 90, 30, 4
    m,n,k = int(lx/a*4.0**(1.0/3)), int(ly/a), int(lz/a)
    #
    bmag = a / 2.0 ** 0.5
    os.system('rm *.cfg *.lmp *.xyz')
    os.system('atomsk --create fcc $a Ni orient 110 -111 1-12 Al_unitcell.cfg')
    os.system('atomsk Al_unitcell.cfg -duplicate $m $n $k Al_supercell.cfg')
    os.system('atomsk Al_supercell.cfg -dislocation 0.51*box 0.51*box edge_rm Z Y $bmag 0.33 data.cfg')
    os.system('atomsk data.cfg -center com final.cfg')
    os.system('atomsk final.cfg lmp')
    #
    WriteDataFile('final.lmp',mass, sys.argv[6])

