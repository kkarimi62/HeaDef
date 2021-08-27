import os

for i in range(8):
	os.system('cp HeaNiCoCrNatom10KTakeOneOutFreezeFract%sRlxd/Run0/scatter.png ./scatter.%s.png'%(i,i))
