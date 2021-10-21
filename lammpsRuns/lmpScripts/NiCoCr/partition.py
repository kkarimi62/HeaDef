import numpy as np
import pandas as pd
import pdb
import sys


def Partition( atoms,box, dmean ):
#    pdb.set_trace()
    #--- grid: tiling mapped box with original size
    (xlin, ylin, zlin), (xv, yv, zv) = lp.GetCubicGrid( box.CellOrigin, 
                                                     box.CellVector, 
                                                     dmean,
                                                     margin = 0.0 * dmean, odd = False )
    xi = np.array(list(zip(xv.flatten(), yv.flatten(), zv.flatten())))
    dvol = (xlin[1]-xlin[0])*(ylin[1]-ylin[0])*(zlin[1]-zlin[0])
    (ny,nx,nz) = xv.shape
    nx -= 1
    ny -= 1
    nz -= 1
    assert nx*ny*nz >= 8, 'decrease division length!'
    print(dmean,nx*ny*nz)

    #--- partition box & assign index to each atom
    wrap = lp.Wrap(atoms,box)
    wrap.WrapCoord() #--- wrap inside
    wrap.Set(atoms)
    assert np.sum(wrap.isInside()) == len(atoms.x)
    wrap.GetDimensionlessCords()
    AtomCellId = (wrap.beta * np.array([nx,ny,nz])).astype(int)
    #--- store in a df
    df = pd.DataFrame(np.c_[pd.DataFrame(atoms.__dict__),AtomCellId],
                         columns=list(pd.DataFrame(atoms.__dict__).keys())+['ix','iy','iz'])
    #--- group & compute p and c
    d = df.groupby(by=['ix','iy','iz']).groups #--- key = cell id, value = list of atoms inside
#     print(len(d))
    assert len(d) == nx*ny*nz, 'empty boxes!'
    keys = d.keys()
#     pdb.set_trace()
    
    #--- output as additional lammps script
    count = 0
    for key in keys:
        sfile=open('ScriptGroup.%s.txt'%count,'w')
        sfile.write('group freeGr id\t')
        atomf = df.iloc[d[key]]
        for i in atomf.id:
            sfile.write('%i\t'%i)
        sfile.write('\ngroup frozGr subtract all freeGr')
        sfile.close()
        count += 1

fileName = sys.argv[1] #'data_init.txt'
dmean = float(sys.argv[2])
pathlib = sys.argv[3]

sys.path.append(pathlib)
import LammpsPostProcess as lp
#
#--- read data file
lmpData = lp.ReadDumpFile( fileName ) 
lmpData.ReadData()
#--- atom obj
atoms = lp.Atoms( **lmpData.coord_atoms_broken[0].to_dict(orient='series') )
#--- box
box = lp.Box( BoxBounds = lmpData.BoxBounds[0],AddMissing = np.array([0.0,0.0,0.0] ) )


#dmean = 10.0
Partition( atoms,box, dmean )