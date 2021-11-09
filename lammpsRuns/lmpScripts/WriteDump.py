import pandas as pd
import sys
import pdb
import numpy as np
import traceback
#
LmpDataFile = sys.argv[1]
ModuFile = sys.argv[2]
WritDump= sys.argv[3]
pathlib = sys.argv[4]
sys.path.append(pathlib)
import LammpsPostProcess as lp


modu = pd.read_csv(ModuFile, sep=' ',header=0)
# 
lmpData = lp.ReadDumpFile( LmpDataFile )
lmpData.ReadData()
#--- atom obj
atoms = lp.Atoms( **lmpData.coord_atoms_broken[0].to_dict(orient='series') )
box = lp.Box( BoxBounds = lmpData.BoxBounds[0],AddMissing = np.array([0.0,0.0,0.0] ) )
#
dfreindx = pd.DataFrame(atoms.__dict__).set_index('id',drop=False) #--- reindex 
#
keys = list(modu.keys()) #list(dfreindx.keys())+list(modu.keys())
#
icel = 0
#if 1:
#try:
#    if 1:
##    while True:
#        Atomid = list(atoms.id) #list( map(int, open('ScriptGroup.%s.txt'%icel).readline().split()[3:]) ) #--- atoms within cell i
#        filtrd = np.c_[dfreindx.loc[Atomid]] #--- filter based on atom id's
# #       pdb.set_trace() 
#        sarr = np.tile(np.c_[modu[modu['#icel']==icel]],(len(filtrd),1)) #--- make copies of mu's: shape(natom,21)
#          
#        arr = np.concatenate((filtrd,sarr),axis=1) #--- shape(natom,cols+21)
#    
#        if icel == 0:
#            arrc = arr.copy()
#        else:
#            arrc = np.concatenate([arrc,arr],axis=0)
#        
#        icel += 1
#except:
#    traceback.print_exc()
#    pass
df = pd.DataFrame( np.c_[modu], columns = keys )
#df = pd.DataFrame( arrc, columns = keys )
atoms = lp.Atoms( **df.to_dict(orient='series'))
wobj = lp.WriteDumpFile(atoms, box)
wobj.Write(WritDump,attrs=keys) #attrs=['id','type','x','y','z']+list(modu.keys())[1:])
#print(df.head())
