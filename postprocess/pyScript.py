
def loadPlane(**kwargs): #--- loading plane

    if 'r' in kwargs and kwargs['r']:
        #--- r plane
        x=unit(np.matmul(h,np.array([1,-2,1,0])))
        y=unit(np.matmul(h,np.array([1,0,1,-1])))
        z=unit(np.matmul(h,np.array([1,0,-1,2]))) #--- load plane

        #--- y has wrong index: just take the cross product
        comp=np.linalg.lstsq(h,np.cross(z,x))[0]
        y=unit(np.matmul(h,comp))

    if 'm' in kwargs and kwargs['m']:
        #--- m plane
#         x=unit(np.matmul(h,np.array([1,-2,1,0])))
#         y=unit(np.matmul(h,np.array([0,0,0,1])))
#         z=unit(np.matmul(h,np.array([1,0,-1,0])))
        x=unit(np.matmul(h,np.array([1,1,-2,0])))
        y=unit(np.matmul(h,np.array([0,0,0,1])))
        z=unit(np.matmul(h,np.array([1,-1,0,0])))

    if 'c' in kwargs and kwargs['c']:
        #--- c plane
        x=unit(np.matmul(h,np.array([1,-2,1,0])))
        y=unit(np.matmul(h,np.array([1,0,-1,0])))
        z=unit(np.matmul(h,np.array([0,0,0,1])))


    if 'a' in kwargs and kwargs['a']:
        #--- a plane
        y=unit(np.matmul(h,np.array([-1,0,1,0])))
        z=unit(np.matmul(h,np.array([1,-2,1,0])))
        x=unit(np.cross(y,z))


    assert np.abs(np.dot(x,y))<1e-6, 'non-orthonal system: x.y=%s'%np.dot(x,y)
    assert np.abs(np.dot(x,z))<1e-6, 'non-orthonal system: x.z=%s'%np.dot(x,z)
    assert np.abs(np.dot(z,y))<1e-6, 'non-orthonal system: z.y=%s'%np.dot(z,y)

    return np.c_[x,y,z]


def slipSys(**kwargs):
#--- input slip system

    if 'basal' in kwargs:
    # #--- basal
        n_plane = unit(np.matmul(h,np.array([0,0,0,1])))
        n_direction = unit(np.matmul(h,np.array([1,-1,0,0])))
        n_direction = unit(np.matmul(h,np.array([1,1,-2,0])))
        n_direction = unit(np.matmul(h,np.array([1,-2,1,0])))


    if 'R_twin_1st' in kwargs: #_10-12' in kwargs:
        n_plane = unit(np.matmul(h,np.array([1,0,-1,2])))
        n_direction = unit(np.matmul(h,np.array([0,-1,1,1])))

    if 'pyramidal' in kwargs:
    # #--- pyramidal dislocation 1/3<>{}
        n_plane = unit(np.matmul(h,np.array([1,1,-2,3])))
        n_direction = unit(np.matmul(h,np.array([-1,1,0,1])))

    if 'R-twin_01-12' in kwargs:
        n_plane = unit(np.matmul(h,np.array([0,1,-1,2])))
#        n_direction = ?


    if 'R_dislocation' in kwargs:
        n_plane = unit(np.matmul(h,np.array([1,0,-1,2])))
        n_direction = unit(np.matmul(h,np.array([1,-2,1,0])))

    if 'junk' in kwargs:
        n_plane = unit(np.matmul(h,np.array([1,-1,0,2])))
        n_direction = unit(np.matmul(h,np.array([-1,1,0,1])))

    if 'prismatic_m' in kwargs:
        n_plane = unit(np.matmul(h,np.array([1,0,-1,0])))
        n_direction = unit(np.matmul(h,np.array([1,-2,1,0])))


    return n_plane, n_direction


GetMag = lambda x: np.sum(x*x)

def GetEig(svect):
    smat=np.array([[svect[0],svect[3],svect[4]],
                   [svect[3],svect[1],svect[5]],
                   [svect[4],svect[5],svect[2]]])
    w, v = LA.eigh(smat)
    smax = 0.5*(w[2]-w[0])
    nvec = v[1]
    return smax

def GetRSS(svect):
    smat=np.array([[svect[0],svect[3],svect[4]],
                   [svect[3],svect[1],svect[5]],
                   [svect[4],svect[5],svect[2]]])
    fvect = np.matmul( smat.T, n_resolve )
    sigma_n = np.dot(fvect,n_resolve)
    sigma_t = GetMag(fvect - sigma_n*n_resolve)
    
        
    return sigma_t


if __name__ == '__main__':
    import LammpsPostProcess as lp
    from numpy import linalg as LA
    import numpy as np
    import sys
    import pandas as pd

    unit = lambda x: x/np.sqrt(np.dot(x,x))

    #--- unit vectors
    e1=np.array([1,0,0])
    e2=np.array([0,1,0])
    e3=np.array([0,0,1])

    #--- lattice basis vectors
    theta=np.pi*30.0/180.0
    a1=np.cos(theta)*e1-np.sin(theta)*e2
    a2=e2
    a3=-(np.cos(theta)*e1+np.sin(theta)*e2)
    a4=e3

    h=np.c_[a1,a2,a3,a4]

    pref = 1e-6 #--- pa to gpa
    itime = 30000

    #--- loading plane
    xyz = loadPlane(a=True)

    #--- slip system
    n_plane, n_direction = slipSys(basal=True)

    n_resolve = np.dot(xyz.T,n_plane)
    t_resolve = np.dot(xyz.T,n_direction)    

	#--- parse lammps dump file
    lmpData = lp.ReadDumpFile( './dump.xyz' ) 
    lmpData.GetCords( ncount = sys.maxsize, sort = False,
                      columns = {'c_spatom[1]':'sxx','c_spatom[2]':'syy','c_spatom[3]':'szz',
                             'c_spatom[4]':'sxy','c_spatom[5]':'sxz','c_spatom[6]':'syz'}
                )

    df = lmpData.coord_atoms_broken[itime]
    stress = np.c_[df[['sxx', 'syy', 'szz','sxy', 'sxz', 'syz']]]
        
    resolved_stress = np.array(list(map(lambda x:GetRSS(x),stress)))*pref

    df_new = pd.DataFrame(np.c_[df['id type x y z'.split()],resolved_stress],columns='id type x y z val'.split())

	#--- output resolved stress
    atom = lp.Atoms(**df_new.to_dict(orient='series'))
    box  = lp.Box( BoxBounds = lmpData.BoxBounds[itime], AddMissing = np.array([0.0,0.0,0.0] ))
    wd = lp.WriteDumpFile(atom, box)
    with open('dumpModified.xyz','w') as output:
	    wd.Write(output,itime=itime,
                 attrs=['id', 'type', 'x', 'y', 'z','val'], 
                 fmt='%i %i %4.3e %4.3e %4.3e %4.3e')
