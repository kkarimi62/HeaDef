
import sys
sys.path.append('/mnt/home/kkarimi/Project/git/HeaDef/postprocess')
import LammpsPostProcess2nd as lp
import numpy as np

rd = lp.ReadDumpFile('data_irradiated.xyz')
rd.GetCords(ncount=sys.maxsize)
keys = list(rd.coord_atoms_broken.keys())
keys.sort()

itime = keys[-1]
atomm = lp.Atoms(**rd.coord_atoms_broken[itime].to_dict(orient='series'))
box=lp.Box(BoxBounds=rd.BoxBounds[itime],AddMissing = np.array([0.0,0.0,0.0] ))
mass={1:1.0,2:1.0,3:1.0}
wd=lp.WriteDataFile(atomm,box,mass)
wd.Write('data_irradiated.dat')
