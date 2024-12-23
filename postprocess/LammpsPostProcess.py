#--- import system modules
import traceback
import sys
import numpy as np
import pandas as pd
import pdb

#--- utility funcs
def ConcatAttr( obj, attrs ):
    indx = 0
    existingAttrs = []
    for attr in attrs:
        if attr in dir(obj):
            existingAttrs += [ attr ]
            if indx == 0:
                XYZ_shifted = obj.__dict__[attr]
            else:
                XYZ_shifted = np.c_[XYZ_shifted,obj.__dict__[attr]]
            indx += 1
    return XYZ_shifted, existingAttrs

def shiftBeta( xyzDimensionless_j, diff ):
    indices = diff > 0.5
    beta0_j = xyzDimensionless_j - 1.0 * indices

    indices = diff < -0.5
    beta0_j += 1.0 * indices
    
    return beta0_j

def GetOrthogonalBasis( CellVector ): 
    a0 = CellVector[:,0]
    l0 = np.dot(a0,a0)**0.5
    n0 = a0 / l0 
    #
    a1 = CellVector[:,1]
    a1_perp = a1 - np.dot( a1, n0 ) * n0
    l1 = np.dot( a1_perp, a1_perp) ** 0.5
    #
    a2 = CellVector[:,2]
    l2 = np.dot(a2,a2)**0.5
    
    return np.c_[a0, a1_perp, a2], [l0, l1, l2]

def GetCubicGrid( CellOrigin, CellVector, dmean, margin, odd=True  ):

    CellVectorOrtho, VectorNorm = GetOrthogonalBasis( CellVector )
    
    volume = np.linalg.det( CellVectorOrtho )


    [nx, ny, nz] = list(map( int, (np.array(VectorNorm)+2*margin) / dmean ))
#    print('nx=',nx)
    if odd:
        if nx%2 == 0:
            nx += 1
        if ny%2 == 0:
            ny += 1
        if nz%2 == 0:
            nz += 1
    
    x = np.linspace( CellOrigin[0] - margin, CellOrigin[0] + VectorNorm[ 0 ] + margin, nx+1,endpoint=True)
    y = np.linspace( CellOrigin[1] - margin, CellOrigin[1] + VectorNorm[ 1 ] + margin, ny+1,endpoint=True)
    z = np.linspace( CellOrigin[2] - margin, CellOrigin[2] + VectorNorm[ 2 ] + margin, nz+1,endpoint=True)

    return (x, y, z), np.meshgrid(x, y,z)

def GetIndex(iy,ix,iz, ny,nx,nz):
    return iy * ( nx * nz ) + ix * nz + iz

def linfunc(x,m,c):
    return m*x+c

def SqError( xdata, ydata):
    z = np.polyfit(np.array(xdata), np.array(ydata), 1)
    return (np.array(ydata)-linfunc(np.array(xdata),*z))**2


############################################################
#######  class ReadDumpFile Reads LAMMPS dump files 
############################################################    
class ReadDumpFile:
    def __init__( self, path, verbose = False ):
        self.path = path
        self.coord_atoms_broken = {}
        self.BoxBounds = {}
        self.verbose = verbose
    
    def GetCords( self, ncount = 1, sort = True, columns = {} ):
        slist = open( self.path )    
        count = 0
        try:
            while True and count <= ncount:
                sarr, cell_vector, itime, cols  = self.GetCordsTimeStep( slist ) #--- get coord
#                 pdb.set_trace()
                #--- insert in a data frame
                self.coord_atoms_broken[ itime ] = pd.DataFrame( sarr.astype('float'), columns = cols )

                #--- cast id and type to 'int'
                self.coord_atoms_broken[ itime ]['id'] = list(map(int,self.coord_atoms_broken[ itime ]['id'].tolist()))[:]
                self.coord_atoms_broken[ itime ]['type'] = list(map(int,self.coord_atoms_broken[ itime ]['type'].tolist()))[:]

                if sort:
                #--- sort
                    self.coord_atoms_broken[ itime ].sort_values( by = 'id', inplace = True )

                #--- reset index
                    self.coord_atoms_broken[ itime ].reset_index( drop=True, inplace=True )
                
                #---
                if len(columns) > 0: #--- change column name
                    self.coord_atoms_broken[ itime ].rename(index=str, columns=columns, inplace = True )

                self.BoxBounds[ itime ] = cell_vector

                count += 1
        except:
            if self.verbose:
                print('reached end of file!')
#            traceback.print_exc()
            pass

    
    def GetCordsTimeStep(self, slist):
        slist.readline()
        itime = int( slist.readline().split()[0] )

        [slist.readline() for i in range(1)]
        nrows = int(slist.readline().split()[0])

        [slist.readline() for i in range(1)]

        CellVector = np.array([slist.readline().split() for i in range( 3 )])

        cols = slist.readline().split()[2:]
#        print(np.array([slist.readline().split() for i in range( nrows )]))

        return np.c_[[slist.readline().split() for i in range( nrows )]], CellVector, itime, cols
    
    def ReadData( self, ncount = 1, columns = {} ):
        itime = 0
        slist = open( self.path )    
        #
        [slist.readline() for i in range(2)]
        #
        natom = int(slist.readline().split()[0])
        ntype = int(slist.readline().split()[0])
        #
        slist.readline()
        #
        cell_vector = np.array([slist.readline().split()[0:2] for i in range( 3 )])
        #
        [slist.readline() for i in range(6+ntype)]
        #       
        sarr = np.array([slist.readline().split()[:5] for i in range( natom )]) #--- get coord

        #--- insert in a data frame
        self.coord_atoms_broken[ itime ] = pd.DataFrame( np.c_[sarr].astype('float'), columns = ['id','type','x','y','z'] )

        #--- cast id and type to 'int'
        self.coord_atoms_broken[ itime ]['id'] = list(map(int,self.coord_atoms_broken[ itime ]['id'].tolist()))[:]
        self.coord_atoms_broken[ itime ]['type'] = list(map(int,self.coord_atoms_broken[ itime ]['type'].tolist()))[:]

        #--- sort
        self.coord_atoms_broken[ itime ].sort_values( by = 'id', inplace = True )

        #--- reset index
        self.coord_atoms_broken[ itime ].reset_index( drop=True, inplace=True )
                
        #---
        if len(columns) > 0: #--- change column name
            self.coord_atoms_broken[ itime ].rename(index=str, columns=columns, inplace = True )

        self.BoxBounds[ itime ] = cell_vector
#         print(self.coord_atoms_broken[ itime ])

############################################################
#######  class WriteDumpFile writes LAMMPS dump files 
############################################################    
class WriteDumpFile:
    def __init__(self, atomm, boxx ):
        self.atom = atomm
        self.box = boxx
        
    def Write(self, outpt, itime=0,attrs=['id', 'type', 'x', 'y', 'z' ], fmt = '%i %i %15.14e %15.14e %15.14e' ):
        natom = len(self.atom.x)
        (xlo,xhi,xy)=list(map(float,self.box.BoxBounds[0,:]))
        (ylo,yhi,junk)=list(map(float,self.box.BoxBounds[1,:]))
        (zlo,zhi,junk)=list(map(float,self.box.BoxBounds[2,:]))
        if type(outpt) == type(' '):
            sfile=open(outpt,'w')
        else:
            sfile = outpt
        sfile.write('ITEM: TIMESTEP\n%d\nITEM: NUMBER OF ATOMS\n%d\nITEM: BOX BOUNDS xy xz yz pp pp pp\n\
                     %15.14e %15.14e %15.14e\n%15.14e\t%15.14e\t%15.14e\n%15.14e\t%15.14e\t%15.14e\nITEM: ATOMS %s\n'\
                     %(itime,natom,xlo,xhi,xy,ylo,yhi,0.0,zlo,zhi,0.0," ".join(map(str,attrs))))

#                     %s %s %s\n%s\t%s\t%s\n%s\t%s\t%s\nITEM: ATOMS id type x y z\n'\

        np.savetxt(sfile,np.c_[pd.DataFrame(self.atom.__dict__)[attrs]],
                   fmt=fmt )

#        for row in :
#            for col in row:
#                sfile.write('%4.3e\t'%col)
#            sfile.write('\n')
#        for idd, typee, x, y, z in zip(self.atom.id, self.atom.type, self.atom.x, self.atom.y, self.atom.z ):
#            sfile.write('%s %s %s %s %s\n'%(int(idd),int(typee),x,y,z))
		  
        if type(outpt) == type(' '):
            sfile.close()

        
############################################################
#######  class WriteDumpFile writes LAMMPS data files 
############################################################    
class WriteDataFile:
    def __init__(self, atomm, boxx, mass ):
        self.atom = atomm
        self.box = boxx
        self.Mass = mass
#        assert len(set(atomm.type)) == len(mass), 'wrong atom types!'
        
    def Write(self, outpt ):
        natom = len(self.atom.x)
        ntype = len(self.Mass)
        (xlo,xhi,xy)=self.box.BoxBounds[0,:]
        (ylo,yhi,junk)=self.box.BoxBounds[1,:]
        (zlo,zhi,junk)=self.box.BoxBounds[2,:]
        sfile=open(outpt,'w')
        sfile.write('LAMMPS Description\n\n%s atoms\n\n%s atom types\n\n\
                     %15.14e %15.14e xlo xhi\n%15.14e %15.14e ylo yhi\n%15.14e %15.14e zlo zhi\n\nMasses\n\n'\
                     %(natom,ntype,float(xlo),float(xhi),float(ylo),float(yhi),float(zlo),float(zhi)))

        for typee in self.Mass: #set(self.atom.type):
            sfile.write('%s %s\n'%(int(typee),self.Mass[typee]))
            
        sfile.write('\nAtoms #molecule-tag atom-type x y z\n\n')
                 
        for idd, typee, x, y, z in zip(self.atom.id, self.atom.type, self.atom.x, self.atom.y, self.atom.z ):
            sfile.write('%s %s %15.14e %15.14e %15.14e\n'%(int(idd),int(typee),x,y,z))
            
        sfile.close()



        

############################################################
#######  class with atom-related attributes 
############################################################    
class Atoms:
    def __init__(self,**kwargs):
#        print 'hello from Atoms const' 
        for attr in kwargs:
            setattr(self,attr, kwargs[attr])
    def __getitem__(self,key):
        return self.__dict__[key]

#     def ____setattr__(self,key,val):
#         print('self.%s=%s'%(key,val))
#         self.__dict__[key] = val
############################################################
#######  class with simulation cell attributes 
############################################################    
class Box:
    def __init__( self, **kwargs ):
        if 'BoxBounds' in kwargs:
            self.BoxBounds = kwargs['BoxBounds']
            self.BasisVectors(**kwargs)
        if 'CellOrigin' in kwargs:
            self.CellOrigin = kwargs['CellOrigin']
        if 'CellVector' in kwargs:
            self.CellVector = kwargs['CellVector']
	#--- 2nd constructor
     
    def BasisVectors( self, **kwargs ):
		#     CellVector[0] = np.c_[CellVector[0],['0.0','0.0','0.0']] #--- ref. state
        if 'AddMissing' in kwargs:
            extraColumn = kwargs['AddMissing']
            if not self.BoxBounds.shape == (3,3):
                self.BoxBounds = np.c_[self.BoxBounds,extraColumn]
#                print('BoxBounds.shape=%s,%s is not (3,3)!'%(self.BoxBounds.shape))
#                print('add %s!'%(extraColumn))

        (xlo, xhi, xy) = list(map( float, self.BoxBounds[ 0 ] )) #--- xlo, xhi, xy
        lx = xhi - xlo - xy
        CellVector0 = np.array( [ lx, 0.0, 0.0 ] )

        (ylo, yhi, xz) =  list(map( float, self.BoxBounds[ 1 ] )) #--- ylo, yhi, xy
        ly = yhi - ylo
        a1 = np.array( [ 0.0, ly, 0.0 ] )
        CellVector1 = CellVector0 * ( xy / lx ) + a1

        (zlo, zhi, yz) =  list(map( float, self.BoxBounds[ 2 ] )) #--- zlo, zhi, xy
        lz = zhi - zlo
        CellVector2 = np.array( [ 0.0, 0.0, lz ] )

        self.CellOrigin = np.array( [ xlo, ylo, zlo ] )
        self.CellVector = np.c_[ CellVector0, CellVector1, CellVector2 ] 

    def SetBoxBounds( self, **kwargs ):
        self.BoxBounds = np.c_[self.CellOrigin, self.CellOrigin + np.matmul(self.CellVector, np.array([1,1,1])), np.array([0,0,0])]


class Wrap():
###########################################################
###### Wrap atom positions 
###########################################################    
    def __init__(self, atoms, box ):
        self.x = atoms.x
        self.y = atoms.y
        self.z = atoms.z
        self.CellVector = box.CellVector
        self.CellOrigin   = box.CellOrigin
  
    def GetDimensionlessCords( self ):
    ############################################################
    ####### compute dimensionless coords of atoms given 
    ####### corresponding cartesian coords
    ############################################################
        xyz_centered = np.c_[self.x,self.y,self.z] - self.CellOrigin
        self.beta = np.matmul( np.linalg.inv(self.CellVector), xyz_centered.T).T

    def WrapCoord( self ):
    ###########################################################
    ###### Wrap atom positions 
    ###########################################################    
        self.GetDimensionlessCords() #--- updates self.beta
        self.beta = self.beta % 1.0
        self.beta = self.beta % 1.0 #--- handle roundoff errors
        self.GetXYZ() #--- updates self.xyz
        indices = self.isInside()
        assert np.all( indices ), 'not all atoms are inside!'

    def GetXYZ( self ):
    ############################################################
    ####### compute coords of atoms given 
    ####### corresponding dimensionless coords
    ############################################################
        XYZ_centered = np.matmul( self.CellVector, self.beta.T ).T #--- xyz in reference state
        xyz = XYZ_centered + self.CellOrigin
        self.x = xyz[:,0]
        self.y = xyz[:,1]
        self.z = xyz[:,2]
        
    def isInside( self ):
        self.GetDimensionlessCords()
        #--- filter: only inside the expanded box 
        beta0 = self.beta[:,0]
        beta1 = self.beta[:,1]
        beta2 = self.beta[:,2]
        index0 = np.all([beta0 >= 0.0, beta0 < 1.0], axis=0 )
        index1 = np.all([beta1 >= 0.0, beta1 < 1.0], axis=0 )
        index2 = np.all([beta2 >= 0.0, beta2 < 1.0], axis=0 )
        return np.all([index0,index1,index2],axis=0)   

    def Set( self, atoms ):
        atoms.x = self.x
        atoms.y = self.y
        atoms.z = self.z

class Map(Wrap):
############################################################
####### map atoms within a tilted box to an orthogonal one
############################################################
    def __init__( self, atoms, box ):
        Wrap.__init__( self, atoms, box ) #--- call parent's constructor                  
    def ChangeBasis( self ):
        self.GetDimensionlessCords() #--- beta0, beta1, beta2
        #--- shift (beta0 + beta1 * dx / |b0| > 1) by - b0
        #--- [b0,b1]^{-1}*(x=|b0|,y)
        b2 = self.CellVector[:,2]
        b1 = self.CellVector[:,1]
        b0 = self.CellVector[:,0]
        norm_b0 = np.dot(b0,b0)**0.5
        shift0 = np.dot(b1, b0 / norm_b0 )
        #
        indices_shiftTrue = (self.beta[:,0]) + (self.beta[:,1])*shift0/norm_b0 >= 1.0
        n = len( indices_shiftTrue )
        shift_matrix =  np.array(indices_shiftTrue).reshape((n,1)) * b0
        #
        self.xm = np.c_[self.x,self.y,self.z] - shift_matrix
        
    def Set( self, atoms ):        
        sdict = {'xm':self.xm[:,0], 'ym':self.xm[:,1], 'zm':self.xm[:,2]}
        atoms.__init__(**sdict)
        

class Copy( Atoms, Wrap ):
############################################################
####### add replicals of the center simulation box
############################################################
    def __init__( self, atoms, box ):
        Atoms.__init__( self, **atoms.__dict__ ) #--- call parent's constructor
        Wrap.__init__( self, atoms, box )
        
    def FullCopies( self ):
        #--- concat. coordinates
        XYZ_shifted, attr0 = ConcatAttr( self, ['x','y','z','xu','yu','zu','xm','ym','zm'] ) #--- add every coord. based attr
        xyz_original = XYZ_shifted.copy()
        assert XYZ_shifted.shape[1] % 3 == 0, 'shifted coordinates must be integer multiple of 3!'
        #
        ID_TYPE_shifted, attr1 = ConcatAttr( self, ['id','type','dx','dy','dz','exy','sxy','sxx','syy','szz','d2min','StructureType','AtomicVolume','C66','C55','C44']) #--- add remaining 'Atoms' attrs
        id_type_original = ID_TYPE_shifted.copy()
		
        #--- cell copies
        for i in [-1,0,1]:
            for j in [-1,0,1]: 
                for k in [-1,0,1]:
        #            print i,j,k
                    if i == j == k == 0:
                        continue
                    #
                    total_shift = np.matmul( self.CellVector, np.array([i,j,k]) ) #--- shape (3,)
                    if XYZ_shifted.shape[1] / 3 == 2:
                        total_shift = np.concatenate([total_shift,total_shift],axis=0) #--- shape: (6,)
                    if XYZ_shifted.shape[1] / 3 == 3:
                        total_shift = np.concatenate([total_shift,total_shift, total_shift],axis=0) #--- shape: (9,)
    #                assert total_shift.shape[ 0 ] == 9
                    #--- add shift
                    xyz_shifted = xyz_original + total_shift
                    #--- append
                    XYZ_shifted = np.concatenate( ( XYZ_shifted, xyz_shifted ), axis = 0 ) #--- bottleneck!!!!!
                    ID_TYPE_shifted = np.concatenate( ( ID_TYPE_shifted, id_type_original ), axis = 0 )
        #--- store full copies in a df
        assert ID_TYPE_shifted.shape[1] == len(attr1)
        assert XYZ_shifted.shape[1] == len(attr0)
        self.df = pd.DataFrame(np.c_[ID_TYPE_shifted,XYZ_shifted],columns=attr1 + attr0)

    def Get( self ):
        return Atoms( **self.df.to_dict(orient = 'list') ) #--- return atom object


    def Expand( self, epsilon = 0.1, mode = 'isotropic'):
    ############################################################
    ####### Get atoms in original box and expand
    ############################################################    
        assert 'xm' in dir(self) and 'ym' in dir(self) and 'zm' in dir(self), 'mapped coordinates are needed!'
        self.FullCopies() #--- full copies
        atomsCopied = self.Get() #--- new atom object

        indices =  self.isInsideExpanded(np.c_[atomsCopied.xm, atomsCopied.ym, atomsCopied.zm], #--- give mapped coordinates
                                         epsilon = epsilon, mode = mode)
        #--- filter!!
        self.df = pd.DataFrame(atomsCopied.__dict__,)[indices]
        
    def isInsideExpanded( self, xyz, epsilon = 0.1, mode = 'isotropic'  ):
    ############################################################
    ####### Get atoms inside an expanded original box
    ############################################################    
        #--- tensor associated with dilation
        identityMat = np.array([[1,0,0],[0,1,0],[0,0,1]])
        if mode == 'isotropic':
	        strainTensor = epsilon * identityMat
        if mode == 'x':
	        strainTensor = epsilon * np.array([[1,0,0],[0,0,0],[0,0,0]])
        #
        #
        CellVectorOrtho, VectorNorm = GetOrthogonalBasis( self.CellVector ) #GetOrthogonalBasis?

        #--- extend diagonal
        rvect = -np.matmul( CellVectorOrtho, np.array([0.5,0.5,0.5]))
        CellOrigin_expanded = self.CellOrigin + np.matmul(strainTensor,rvect)

        #--- extend basis vectors
        CellVector_expanded = np.matmul( identityMat + strainTensor, CellVectorOrtho )

        
        #--- new atom, box, and wrap object
        atoms = Atoms(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2])
        box = Box( CellOrigin = CellOrigin_expanded, CellVector = CellVector_expanded )
        wrap = Wrap( atoms, box )
        
        return wrap.isInside()

class Compute( Atoms, Box ):
############################################################
####### base class for computing atom properties
############################################################
    def __init__( self, atoms, box ): #--- call parent constructor 
        Atoms.__init__(self, **atoms.__dict__) 
        Box.__init__(self, CellOrigin = box.CellOrigin, CellVector = box.CellVector )

    def Get( self, attrs = [] ):
    ############################################################
    ####### get function returning an atom object
    ############################################################
        assert np.all(np.array( [ item in self.__dict__ for item in attrs ] )), 'not all attributes are available!'
#        assert np.all(map((self.__dict__).has_key, attrs )), 'not all attributes are available!'
        values = list(map(self.__dict__.get,attrs))
        df = pd.DataFrame(np.c_[values].T, columns = attrs )
        return Atoms(**df.to_dict(orient = 'list ') )

    def Set( self, value, attrs=[]):
    ############################################################
    ####### set function calling atom class constructor
    ############################################################
#        pdb.set_trace()
        Atoms.__init__( self, **pd.DataFrame(value,columns=attrs).to_dict(orient='list'))
#		self.=value[:,0]


class ComputeD2min( Compute ):
    def __init__( self, atoms, box, delx ):
        Compute.__init__( self, atoms, box )
        self.delx = delx

    def Partition( self ):
        #--- cubic grid
        (xlin, ylin, zlin), (xv, yv, zv) = GetCubicGrid( self.CellOrigin, 
                                                         self.CellVector, 
                                                         self.delx,
                                                         margin = 0.0 )        #--- assign index
         #--- set bounds
        (ny,nx,nz) = xv.shape
        xlo, ylo, zlo = np.min(xlin), np.min(ylin), np.min(zlin)
        xhi, yhi, zhi = np.max(xlin), np.max(ylin), np.max(zlin)
        lx, ly, lz = xhi-xlo, yhi-ylo, zhi-zlo 
        
        ix = (nx*(np.c_[self.xm]-xlo)/lx).astype(int).flatten() #--- type  self.xm???
        assert np.all([ix>=0,ix<nx])

        iy = (ny*(np.c_[self.ym]-ylo)/ly).astype(int).flatten()
        assert np.all([iy>=0,iy<ny])

        iz = (nz*(np.c_[self.zm]-zlo)/lz).astype(int).flatten()
        assert np.all([iz>=0,iz<nz])

        self.blockid = GetIndex(iy,ix,iz, ny,nx,nz)
        (self.ny,self.nx,self.nz) = (ny,nx,nz)

    def D2min( self ):
        #--- loop over partitions and compute F_{\alpha\beta}=\partial u_\alpha/x_\beta
        natoms = len( self.xm )
        d2min = np.zeros( natoms * 9 ).reshape((natoms,9))
        natoms0 = natoms
        natoms = 0
        
        for indx in range(self.ny*self.nx*self.nz):
    
            #--- filtering
            atomi = Atoms(**pd.DataFrame(np.c_[self.id,self.type,self.x,self.y,self.z,self.xm,self.ym,self.zm,self.dx,self.dy,self.dz],
                    columns = ['id','type','x','y','z','xm','ym','zm','dx','dy','dz'])[self.blockid == indx].to_dict(orient='list'))
        

            natom = len( atomi.xm )
            if natom == 0:
                continue


            #--- deformation gradients
            D2min =  SqError(atomi.xm, atomi.dx)
            D2min += SqError(atomi.ym, atomi.dx)
            D2min += SqError(atomi.zm, atomi.dx)
            D2min += SqError(atomi.xm, atomi.dy)
            D2min += SqError(atomi.ym, atomi.dy)
            D2min += SqError(atomi.zm, atomi.dy)
            D2min += SqError(atomi.xm, atomi.dz)
            D2min += SqError(atomi.ym, atomi.dz)
            D2min += SqError(atomi.zm, atomi.dz)

            #--- store
            d2min[ natoms : natoms + natom ] = np.c_[atomi.id,atomi.type,atomi.x,atomi.y,atomi.z,atomi.xm,atomi.ym,atomi.zm,D2min]

            natoms += natom
        assert natoms == natoms0
        assert len(set(d2min[:,0])) == len( d2min ), 'boxes are overlapping!'
        
        
                          
        self.Set( d2min, attrs=['id','type','x','y','z','xm','ym','zm','d2min'])
        
	
# class ComputeRdf( Compute, Wrap ):
# ############################################################
# ####### compute radial pair correlation function
# ####### in a periodic system
# ############################################################
#     def __init__( self, atoms, box, cutoff = 1.0, NMAX = 1000):#, 
# #                 n_neigh_per_atom = 20):
#         Compute.__init__( self, atoms, box )
#         Wrap.__init__( self, atoms, box )
        
#         self.cutoff = cutoff
#         self.NMAX = NMAX
        
#         #--- number density
#         CellVectorOrtho, VectorNorm = GetOrthogonalBasis( self.CellVector )
#         volume = np.linalg.det( CellVectorOrtho )
#         self.rho = len( self.x ) / volume
        
#         self.n_neigh_per_atom = 2 * int( self.rho * self.cutoff * self.cutoff * self.cutoff * 4.0 * np.pi / 3.0 )
        
        
#     def GetXYZ( self ): #--- overwrite GetXYZ in Wrap
#     ############################################################
#     ####### compute coords of atoms given 
#     ####### corresponding dimensionless coords
#     ############################################################
#         return np.matmul( self.CellVector, self.beta.T ).T #--- xyz in reference state

    
#     def Distance( self, WRAP = True , **kwargs ):
#         self.GetDimensionlessCords() #--- dimensionless cords
#         eta = self.beta 
#         #---    
#         nmax = min(self.NMAX, len( self.x ))
#         i = 0
#         nr = 0
#         self.rlist = np.zeros(nmax*self.n_neigh_per_atom)
#         #--- filter center particle
#         filtr = np.ones(len(self.x),dtype=int) * True
#         if 'FilterCenter' in kwargs:
#             filtr = kwargs['FilterCenter']
#         #
#         kount = 0
#         while i < nmax: #--- pair-wise dist: i is the center atom index
#             if not filtr[i]: 
#                 i += 1
#                 continue
#             #--- distance matrix
#             df_dx = eta[ i+1:,0 ] - eta[ i, 0 ] #--- avoid double counting
#             df_dy = eta[ i+1:,1 ] - eta[ i, 1 ]
#             df_dz = eta[ i+1:,2 ] - eta[ i, 2 ]
#             if WRAP: #--- pbc effects
#                 df_dx -= (df_dx > 0.5 )*1
#                 df_dx += (df_dx < - 0.5 )*1
#                 df_dy -= (df_dy > 0.5 )*1
#                 df_dy += (df_dy < - 0.5)*1
#                 df_dz -= (df_dz > 0.5)*1
#                 df_dz += (df_dz < - 0.5)*1

                
#             self.beta = np.c_[df_dx,df_dy,df_dz] #--- relative dimensionless coordinates
#             #--- distance vector
#             disp_vector = self.GetXYZ() 
#             disp2 = disp_vector * disp_vector
#             #--- distance
#             df_sq = ( disp2[:,0]+disp2[:,1]+disp2[:,2] ) ** 0.5 
#             df_sq = df_sq[ df_sq < self.cutoff ] #--- filtering        
#             #--- concatenate
#             assert nr+len(df_sq) <= self.rlist.shape[0], '%s, %s increase buffer size!'%(nr+len(df_sq),self.rlist.shape[0])
#             self.rlist[nr:nr+len(df_sq)] = df_sq
#             #---
#             kount += 1
#             i += 1
#             nr += len( df_sq )
#         self.NMAX=kount
#         print(self.NMAX)

#     def PairCrltn( self, nbins = 32, **kwargs ):
#         #--- histogram
#         slist = self.rlist[self.rlist>0]
#         rmin = slist.min()
#         rmax = slist.max()
        
#         volume = 4.0*np.pi*rmax**3/3
#         self.rho = (len( slist ) + 1) / volume

#         #ndecades =  int(np.ceil(np.log10(rmax/rmin)))
#         if 'bins' in kwargs:
#             bins=kwargs['bins']
#         else:
#             ßbins = np.linspace(rmin,rmax,nbins) #np.logspace(np.log10(rmin),np.log10(rmax),ndecades*4)
#         hist, bin_edges = np.histogram( slist, bins = bins) #, density=True  ) #--- normalized g(r)
# #         print(bin_edges)
#         dr = bin_edges[1]-bin_edges[0]
        
        
# #        rmean, bin_edges = np.histogram( slist, bins = bins, weights = slist ) #--- \sum r_i
#         count, bin_edges = np.histogram( slist, bins = bins ) #--- n_i
# #         rmean /= count #--- average distance: \sum r_i/n_i
#         rmean=0.5*(bin_edges[:-1]+bin_edges[1:])



#         hist = hist.astype(float)
# #        hist *= len( slist ) #--- 
# #        pdb.set_trace()
#         hist /= 4*np.pi*rmean*rmean*dr #self.NMAX
#         hist /= self.rho    
        
#         self.rmean = rmean
#         self.hist = hist
#         self.err = hist / count ** 0.5

#     def Get( self ):
#         return self.rmean, self.hist, self.err

class ComputeRdf( Compute, Wrap ):
############################################################
####### compute radial pair correlation function
####### in a periodic system
############################################################
    def __init__( self, atoms, box, cutoff = 1.0, NMAX = 1000):#, 
#                 n_neigh_per_atom = 20):
        Compute.__init__( self, atoms, box )
        Wrap.__init__( self, atoms, box )
        
        self.cutoff = cutoff
        self.NMAX = NMAX
        
        #--- number density
        CellVectorOrtho, VectorNorm = GetOrthogonalBasis( self.CellVector )
        volume = np.linalg.det( CellVectorOrtho )
        self.rho = len( self.x ) / volume
        
        self.n_neigh_per_atom = 2 * int( self.rho * self.cutoff * self.cutoff * self.cutoff * 4.0 * np.pi / 3.0 )
        
        
    def GetXYZ( self ): #--- overwrite GetXYZ in Wrap
    ############################################################
    ####### compute coords of atoms given 
    ####### corresponding dimensionless coords
    ############################################################
        return np.matmul( self.CellVector, self.beta.T ).T #--- xyz in reference state

    
    def Distance( self, WRAP = True , **kwargs ):
        self.GetDimensionlessCords() #--- dimensionless cords
        eta = self.beta 
        #---    
        nmax = min(self.NMAX, len( self.x ))
        i = 0
        nr = 0
        self.rlist = np.zeros(nmax*self.n_neigh_per_atom)
        #--- filter center particle
        filtr = np.ones(len(self.x),dtype=int) * True
        if 'FilterCenter' in kwargs:
            filtr = kwargs['FilterCenter']
        #
        kount = 0
        while i < nmax: #--- pair-wise dist: i is the center atom index
            if not filtr[i]: 
                i += 1
                continue
            #--- distance matrix
            df_dx = eta[ i+1:,0 ] - eta[ i, 0 ] #--- avoid double counting
            df_dy = eta[ i+1:,1 ] - eta[ i, 1 ]
            df_dz = eta[ i+1:,2 ] - eta[ i, 2 ]
            if WRAP: #--- pbc effects
                df_dx -= (df_dx > 0.5 )*1
                df_dx += (df_dx < - 0.5 )*1
                df_dy -= (df_dy > 0.5 )*1
                df_dy += (df_dy < - 0.5)*1
                df_dz -= (df_dz > 0.5)*1
                df_dz += (df_dz < - 0.5)*1

                
            self.beta = np.c_[df_dx,df_dy,df_dz] #--- relative dimensionless coordinates
            #--- distance vector
            disp_vector = self.GetXYZ() 
            disp2 = disp_vector * disp_vector
            #--- distance
            df_sq = ( disp2[:,0]+disp2[:,1]+disp2[:,2] ) ** 0.5 
            df_sq = df_sq[ df_sq < self.cutoff ] #--- filtering        
            #--- concatenate
            assert nr+len(df_sq) <= self.rlist.shape[0], '%s, %s increase buffer size!'%(nr+len(df_sq),self.rlist.shape[0])
            self.rlist[nr:nr+len(df_sq)] = df_sq
            #---
            kount += 1
            i += 1
            nr += len( df_sq )
        self.NMAX=kount
        print(self.NMAX)

    def PairCrltn( self, nbins = 32, **kwargs ):
        #--- pairwise distances
        if 'rlist' in kwargs:
            self.rlist = kwargs['rlist']
            
        #--------------    
        #--- histogram
        #--------------
        slist = self.rlist[self.rlist>0]
        rmin = slist.min()
        rmax = slist.max()
        
        #--- density
        volume = 4.0*np.pi*rmax**3/3 
        self.rho = (len( slist ) + 1) / volume

        if 'bins' in kwargs:
            bins=kwargs['bins']
        else:
            bins = np.linspace(rmin,rmax,nbins) #np.logspace(np.log10(rmin),np.log10(rmax),ndecades*4)

        rmean, bin_edges = np.histogram( slist, bins = bins, weights = slist ) #--- \sum r_i
        hist, bin_edges = np.histogram( slist, bins = bins) #, density=True  ) #--- normalized g(r)


        #--- normalize
        count = hist.copy()
        #--- mean r
        dr = bin_edges[1]-bin_edges[0]
        if 'regular_r' in kwargs and kwargs['regular_r']:
            rmean=0.5*(bin_edges[:-1]+bin_edges[1:])
        else:
            rmean /= count #--- average distance: \sum r_i/n_i
        hist = hist.astype(float)
        hist /= 4*np.pi*rmean*rmean*dr
        hist /= self.rho    
        
        #--- set
        self.rmean = rmean
        self.hist = hist
        self.count = count
        self.err = hist / count ** 0.5
        
        
    def Sro(self,neigh_list,typei,typej,bins):
        '''
            Warren-Cowley order parameter
        '''
        filtr_i = neigh_list['type']==typei #--- filter based on center atom i
        self.PairCrltn(  
                        bins=bins, 
                        rlist=neigh_list[filtr_i].DIST )
        ntot = self.count.copy()

        
        #--- filter based on pair ij
        filtr_ij = np.all([neigh_list['type']==typei,neigh_list['Jtype'].astype(int)==typej],axis=0)
        self.PairCrltn(  
                  bins=bins, 
                  rlist=neigh_list[filtr_ij].DIST )
        nij = self.count.copy()
        
        return self.rmean, nij/ntot, (1.0/nij**0.5-1.0/ntot**0.5)*nij/ntot

    @staticmethod 
    def sroPerAtom(center_atom_id,sdict,neigh_list,Typej):
        indices = sdict[ center_atom_id ]
        neigh_filtrd = neigh_list.iloc[indices]
        ntot = len(neigh_filtrd)

        #--- filter based on pair ij
        nij = np.zeros(len(Typej))
        for typej,indx in zip(Typej,range(len(Typej))):
            filtr_ij = neigh_filtrd['Jtype'].astype(int)==typej
            nij[indx] = len(neigh_filtrd[filtr_ij])

        return nij/ntot if ntot != 0 else np.nan


    def AtomWiseSro(self,neigh_list,typej):
        '''
           atom-wise Warren-Cowley order parameter
        '''

        sdict = neigh_list.groupby('id').groups
        keys = list(sdict.keys())
        keys.sort()
        sro = list(map(lambda x: ComputeRdf.sroPerAtom(x,sdict,neigh_list,typej),keys))

        return pd.DataFrame(np.c_[keys,sro],columns=('id %s %s %s'%tuple(typej)).split())

    def Get( self ):
        return self.rmean, self.hist, self.err

class ComputeCrltn( ComputeRdf ):
############################################################
####### compute CrltnFunc for data defined over 
####### an unstructured grid point
############################################################    
    def __init__( self, atoms, box, val, 
                 cutoff = 1.0, dx = 1.0,
                 NMAX = 1000):
        ComputeRdf.__init__( self, atoms, box, cutoff = cutoff, NMAX = NMAX)
     
        #--- zscore values
        self.value = val - np.mean(val)
        self.value /= np.std(self.value)
        self.dx = dx #--- discretization length: 1st peak in rdf


    
    def Distance( self, WRAP = True ):
        self.GetDimensionlessCords() #--- dimensionless cords
        eta = self.beta 
        #---    
        nmax = min(self.NMAX, len( self.x ))
        i = 0
        nr = 0
        self.rlist = np.zeros(nmax*self.n_neigh_per_atom)
        self.rvect = np.zeros(nmax*self.n_neigh_per_atom*3).reshape((nmax*self.n_neigh_per_atom,3))
        self.flist = np.zeros(nmax*self.n_neigh_per_atom)
    
    
        while i < nmax: #--- pair-wise dist.
            #--- distance matrix
            df_dx = eta[ i+1:,0 ] - eta[ i, 0 ] #--- avoid double counting
            df_dy = eta[ i+1:,1 ] - eta[ i, 1 ]
            df_dz = eta[ i+1:,2 ] - eta[ i, 2 ]
            product = self.value[i+1:]*self.value[i]
            if WRAP: #--- pbc effects
                df_dx -= (df_dx > 0.5 )*1
                df_dx += (df_dx < - 0.5 )*1
                df_dy -= (df_dy > 0.5 )*1
                df_dy += (df_dy < - 0.5)*1
                df_dz -= (df_dz > 0.5)*1
                df_dz += (df_dz < - 0.5)*1

                
            self.beta = np.c_[df_dx,df_dy,df_dz] #--- relative dimensionless coordinates
            #--- distance vector
            disp_vector = self.GetXYZ() 
            disp2 = disp_vector * disp_vector
            #--- distance
            df_sq = ( disp2[:,0]+disp2[:,1]+disp2[:,2] ) ** 0.5 
            #--- filtering
            disp_vector = disp_vector[ df_sq < self.cutoff ]
            product = product[ df_sq < self.cutoff ]
            df_sq = df_sq[ df_sq < self.cutoff ]    
            #--- concatenate
            assert nr+len(df_sq) <= self.rlist.shape[0], '%s, %s increase buffer size!'%(nr+len(df_sq),self.rlist.shape[0])
            self.rlist[nr:nr+len(df_sq)] = df_sq
            self.rvect[nr:nr+len(df_sq)] = disp_vector
            self.flist[nr:nr+len(df_sq)] = product.flatten()
        
        #---
            i += 1
            nr += len( df_sq )

    def AutoCrltn( self, RADIAL = True ):
        self.RADIAL = RADIAL
        #--- histogram
        slist = self.rlist[self.rlist>0]
        self.rvect = self.rvect[self.rlist>0]
        self.flist = self.flist[self.rlist>0]
        rmin = slist.min()
        rmax = slist.max()

        if RADIAL:
            nbin = int((rmax-rmin)/self.dx)
            bins = np.linspace(rmin,rmax,nbin) #np.logspace(np.log10(rmin),np.log10(rmax),ndecades*4)
            self.fmean, bin_edges = np.histogram( slist, bins = bins, weights = self.flist ) #--- \sum f_i.fj
            self.rmean, bin_edges = np.histogram( slist, bins = bins, weights = slist ) #--- \sum r_i
            self.count, bin_edges = np.histogram( slist, bins = bins ) #--- n_i
            #
            self.rmean /= self.count #--- average distance: \sum r_i/n_i
            self.fmean /= self.count

        else: #--- 3d correlations
            xmin, xmax = self.rvect[:,0].min(), self.rvect[:,0].max()
            ymin, ymax = self.rvect[:,1].min(), self.rvect[:,1].max()
            zmin, zmax = self.rvect[:,2].min(), self.rvect[:,2].max()
            nbinx = int((xmax-xmin)/self.dx)
            nbiny = int((ymax-ymin)/self.dx)
            nbinz = int((zmax-zmin)/self.dx)
            bins_yxz = ( np.linspace( ymin, ymax, nbiny + 1, endpoint = True ), \
                         np.linspace( xmin, xmax, nbinx + 1, endpoint = True ), \
                         np.linspace( zmin, zmax, nbinz + 1, endpoint = True ) )

            #--- append negative r and corresponding f (c is hermitian)
            self.rvect = np.concatenate((self.rvect,-self.rvect),axis=0)
            self.flist = np.concatenate((self.flist,self.flist),axis=0)

            #--- swap columns
            rxcol = self.rvect[:,0]
            rycol = self.rvect[:,1]
            rzcol = self.rvect[:,2]
            self.rvect = np.c_[ rycol, rxcol, rzcol ]

            #--- histograms
            self.fmean, bin_edges = np.histogramdd( self.rvect, bins = bins_yxz, weights = self.flist ) #--- 3d histogram
            self.rx, bin_edges = np.histogramdd( self.rvect, bins = bins_yxz, weights = rxcol ) #--- \sum r_i
            self.ry, bin_edges = np.histogramdd( self.rvect, bins = bins_yxz, weights = rycol ) #--- \sum r_i
            self.rz, bin_edges = np.histogramdd( self.rvect, bins = bins_yxz, weights = rzcol ) #--- \sum r_i
            self.count, bin_edges = np.histogramdd( self.rvect, bins = bins_yxz ) #--- n_i


            #---- zero count????
            self.count[self.count==0] = 1
            self.rx /= self.count 
            self.ry /= self.count 
            self.rz /= self.count 
            self.fmean /= self.count
#            pdb.set_trace()

    def Get( self ):
        if self.RADIAL:
            return self.rmean, self.fmean, 1/self.count**0.5
        else:
            return self.rx, self.ry, self.rz,  self.fmean, 1/self.count**0.5


class ComputeDisp( Compute, Wrap ):
############################################################
####### compute atoms' displacements given their
####### reference state
############################################################
    def __init__( self, atoms, box, atoms0, box0 ): #--- '0' denotes ref. state
        Compute.__init__( self, atoms, box ) #--- call parent's constructor        
        Wrap.__init__( self, atoms, box )
        self.atoms0 = atoms0
        self.box0 = box0
            
    def SetUnwrapped( self ):
        #--- only if unwrapped coords are available
        assert 'xu' in dir(self) and 'yu' in dir(self) and 'zu' in dir(self) and\
               'xu' in dir(self.atoms0) and 'yu' in dir(self.atoms0) and 'zu' in dir(self.atoms0),\
               'unwrapped coordinates are needed!'
       #--- displacement: r^{unwrpd}_j - r^{wrpd}_i
        disp = np.c_[self.xu,self.yu,self.zu] - np.c_[self.atoms0.xu,self.atoms0.yu,self.atoms0.zu]
        self.atoms0.dx = disp[:,0]
        self.atoms0.dy = disp[:,1]
        self.atoms0.dz = disp[:,2]
        #--- 
        
    def SetWrapped( self ):
        self.EstimateUnwrappedCord()    
        #--- displacement: r^{unwrpd}_j - r^{wrpd}_i
        disp = np.c_[self.x,self.y,self.z] - np.c_[self.atoms0.x,self.atoms0.y,self.atoms0.z]
        self.atoms0.dx = disp[:,0]
        self.atoms0.dy = disp[:,1]
        self.atoms0.dz = disp[:,2]
        print('warning: attributes x, y, z are now unwrapped!')
        #--- 
   
    def Get( self, attrs = [] ): #--- overwrite the base function
#        pdb.set_trace()
        assert np.all(np.array( [ item in self.atoms0.__dict__ for item in attrs ] )), 'not all attributes are available!'
#        assert np.all(map((self.atoms0.__dict__).has_key, attrs )), 'not all attributes are available!'
        values = list(map(self.atoms0.__dict__.get,attrs))
        df = pd.DataFrame(np.c_[values].T, columns = attrs )
        return Atoms(**df.to_dict(orient = 'list ') )
 
    def EstimateUnwrappedCord( self ):
        #--- dimensionless cords
        self.GetDimensionlessCords() #--- current frame
        #
        wrap0 = Wrap(self.atoms0, self.box0)
        wrap0.GetDimensionlessCords() #--- reference frame
        
        #--- shift to get unwrapped cords
        diff = self.beta - wrap0.beta

        #--- new dimensionless cords
        beta0_j = shiftBeta( self.beta[:,0], diff[:,0])
        beta1_j = shiftBeta( self.beta[:,1], diff[:,1])
        beta2_j = shiftBeta( self.beta[:,2], diff[:,2])
        self.beta = np.c_[beta0_j,beta1_j,beta2_j]
        
        #--- unwrapped cords at deformed state
        self.GetXYZ()

class ComputeStrn( Compute ):
############################################################
####### compute atomistic strains 
############################################################
    def __init__( self, atoms, box ):
        Compute.__init__( self, atoms, box )
    
    def Reshape( self, xlin, ylin, zlin ):
    #--- reshape matrix
        nx,ny,nz = len(xlin), len(ylin),len(zlin)
#         lx = xlin[-1]-xlin[0]
#         ly = ylin[-1]-ylin[0]
#         lz = zlin[-1]-zlin[0]
        self.ux = np.c_[self.dx].reshape((ny,nx,nz))
        self.uy = np.c_[self.dy].reshape((ny,nx,nz))
        self.uz = np.c_[self.dz].reshape((ny,nx,nz))
        self.bins = (xlin, ylin, zlin)
        
    def Gradient( self ):
        (xlin, ylin, zlin) = self.bins
        #
        self.ux_x = np.gradient(self.ux,xlin,axis=1,edge_order=2).flatten()
        self.ux_y = np.gradient(self.ux,ylin,axis=0,edge_order=2).flatten()
        self.ux_z = np.gradient(self.ux,zlin,axis=2,edge_order=2).flatten()
        #
        self.uy_x = np.gradient(self.uy,xlin,axis=1,edge_order=2).flatten()
        self.uy_y = np.gradient(self.uy,ylin,axis=0,edge_order=2).flatten()
        self.uy_z = np.gradient(self.uy,zlin,axis=2,edge_order=2).flatten()
        #
        self.uz_x = np.gradient(self.uz,xlin,axis=1,edge_order=2).flatten()
        self.uz_y = np.gradient(self.uz,ylin,axis=0,edge_order=2).flatten()
        self.uz_z = np.gradient(self.uz,zlin,axis=2,edge_order=2).flatten()
        
#             #--- gradient
#         if method == 'diff':
#             appendd=ux[:,0,:].reshape((ny,1,nz))
#             ux_x = np.diff(ux, axis=1,append=appendd)
#             appendd=ux[0,:,:].reshape((1,nx,nz))
#             ux_y = np.diff(ux, axis=0,append=appendd)
#             appendd=ux[:,:,0].reshape((ny,nx,1))
#             ux_z = np.diff(ux, axis=2,append=appendd)

#             appendd=uy[:,0,:].reshape((ny,1,nz))
#             uy_x = np.diff(uy, axis=1,append=appendd)
#             appendd=uy[0,:,:].reshape((1,nx,nz))
#             uy_y = np.diff(uy, axis=0,append=appendd)
#             appendd=uy[:,:,0].reshape((ny,nx,1))
#             uy_z = np.diff(uy, axis=2,append=appendd)

#             appendd=uz[:,0,:].reshape((ny,1,nz))
#             uz_x = np.diff(uz, axis=1,append=appendd)
#             appendd=uz[0,:,:].reshape((1,nx,nz))
#             uz_y = np.diff(uz, axis=0,append=appendd)
#             appendd=uz[:,:,0].reshape((ny,nx,1))
#             uz_z = np.diff(uz, axis=2,append=appendd)

#         if method == 'fft':
#             ux_x = GetDerivX( ux,lx ) 
#             ux_y = GetDerivY( ux,lx ) 
#             ux_z = GetDerivZ( ux,lx ) 

#             uy_x = GetDerivX( uy,ly ) 
#             uy_y = GetDerivY( uy,ly ) 
#             uy_z = GetDerivZ( uy,ly ) 

#             uz_x = GetDerivX( uz,lz ) 
#             uz_y = GetDerivY( uz,lz ) 
#             uz_z = GetDerivZ( uz,lz ) 

    def SetStrn( self, component ):
        if component == 'exx':
            self.exx = self.ux_x
            self.exx -= np.mean(self.exx)
        if component == 'exy':
            self.exy = self.eyx = 0.5 * ( self.ux_y + self.uy_x )
            self.exy -= np.mean(self.exy)
            self.eyx -= np.mean(self.eyx)
        if component == 'exz':
            self.exz = self.ezx = 0.5 * ( self.ux_z + self.uz_x )
            self.exz -= np.mean(self.exz)
            self.ezx -= np.mean(self.ezx)
        if component == 'eyy':
            self.eyy = self.uy_y
            self.eyy -= np.mean(self.eyy)
        if component == 'eyz':
            self.eyz = self.ezy = 0.5 * ( self.uy_z + self.uz_y )
            self.eyz -= np.mean(self.eyz)
            self.ezy -= np.mean(self.ezy)
        if component == 'ezz':
            self.ezz = self.uz_z
            self.ezz -= np.mean(self.ezz)

#    def GetStrn( self, attrs = [] ):
#		assert np.all(map((self.__dict__).has_key, attrs )), 'not all attributes are available!'
#		values = map(self.__dict__.get,attrs)
#		df = pd.DataFrame(np.c_[values].T, columns = attrs )
#		return Atoms(**df.to_dict(orient = 'list ') )


if __name__ in '__main__':

#     fileName = '/Users/Home/Desktop/Tmp/txt/git/CrystalPlasticity/BmgData/FeNi_glass.dump'
#     myRDF = ReadDumpFile( fileName )
#     myRDF.GetCords( ncount = sys.maxint )

#     #
#     myAtoms = Atoms( **myRDF.coord_atoms_broken[0].to_dict(orient='list') )
#     #
#     pdb.set_trace()
#     myBox = Box( BoxBounds = myRDF.BoxBounds[0] )
#     myBox.BasisVectors()
    
    n=1000
    xyz = np.random.random((n,3)) 
    atom_tmp = Atoms(**pd.DataFrame(np.c_[np.arange(n),np.ones(n),xyz],
                                       columns=['id','type','x','y','z']).to_dict(orient='list'))
    box_tmp = Box(BoxBounds=np.array([[1,0,0],[0,1,0],[0,0,1]]))
    
    wdf = WriteDumpFile(atom_tmp,box_tmp)
    wdf.Write('junk.xyz')
    
#--- test on random data
#     n=1000
#     xyz = np.random.random((n,3)) 
#     atom_tmp = Atoms(**pd.DataFrame(np.c_[np.arange(n),np.ones(n),xyz],
#                                        columns=['id','type','x','y','z']).to_dict(orient='list'))
#     box_tmp = lp.Box(CellOrigin=np.array([0,0,0]),CellVector=np.array([[1,0,0],[0,1,0],[0,0,1]]))
#     val = np.random.random(n) #np.sin(2*np.pi*xyz[:,0])
#     crltn = ComputeCrltn(    atom_tmp, box_tmp,
#                                  val,
#                                  cutoff=1.0*3**.5, dx=0.05,
#                                  NMAX = n, n_neigh_per_atom = 10000,
#                          )
#     crltn.Distance()
#     crltn.AutoCrltn(RADIAL = True)
#     bin_edges,  hist, err = crltn.Get()
