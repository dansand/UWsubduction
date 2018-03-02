
import numpy as np
import underworld as uw
from underworld import function as fn
from mpi4py import MPI
comm = MPI.COMM_WORLD
import operator

from scipy.spatial import cKDTree as kdTree

class interface2D(object):
    """
    All the bits and pieces needed to define an interface(in 2D) from a string of points
    """


    def __init__(self, mesh, velocityField, pointsX, pointsY, fthickness, fID, insidePt=(0.0,0.0)):


        # Interface swarms are probably sparse, and on most procs will have nothing to do
        # if there are no particles (or not enough to compute what we need)
        # then set this flag and return appropriately. This can be checked once the swarm is
        # populated.

        #however, a major constraint is that in order to get local + shadow particles,
        #we rely to a UW function with an mpi barrier (i.e. swarm.shadow_particles_fetch())

        #We supply a @property method self.data as a handle for the local + shadow particles
        #the main thing to remember is to not make calls to self.data in any kind of conditional setting
        #in which some processors might be excluded.

        self.empty = False

        # Should do some checking first

        self.mesh = mesh
        self.velocity = velocityField
        self.thickness = fthickness
        self.ID = fID
        self.insidePt = insidePt
        self.director = None

        # Set up the swarm and variables on all procs

        self.swarm = uw.swarm.Swarm( mesh=self.mesh, particleEscape=True )
        self.director = self.swarm.add_variable( dataType="double", count=2)
        self._swarm_advector = uw.systems.SwarmAdvector( swarm=self.swarm,
                                                         velocityField=self.velocity, order=2 )

        self.swarm.add_particles_with_coordinates(np.stack((pointsX, pointsY)).T)
        self.director.data[...] = 0.0

        self._update_kdtree()
        self._update_surface_normals()

        return

    @property
    def data(self):
        #this getter must be called by all procs in parallel
        #https://github.com/underworldcode/underworld2/blob/12a090589d1daaffddd685678d7966e4c664aeab/underworld/swarm/_swarm.py#L536
        return self.all_coords()

    def add_points(self, pointsX, pointsY):

        self.swarm.add_particles_with_coordinates(np.stack((pointsX, pointsY)).T)

        self.rebuild()


    def rebuild(self):

        self._update_kdtree()
        self._update_surface_normals()

        return


    def _update_kdtree(self):

        self.empty = False
        #all_particle_coords = self.data

#        self.swarm.shadow_particles_fetch()

#        dims = self.swarm.particleCoordinates.data.shape[1]

#        pc = np.append(self.swarm.particleCoordinates.data,
#                       self.swarm.particleCoordinates.data_shadow)

#        all_particle_coords = pc.reshape(-1,dims)

        all_particle_coords = self.data

        if len(all_particle_coords) < 3:
            self.empty = True
            #self.kdtree = lambda x: float('inf')
            #trying this instead,
            self.kdtree = kdTree(np.empty((2,0)))
        else:
            self.kdtree = kdTree(all_particle_coords)

        return


    def advection(self, dt):
        """
        Update interface swarm particles as material points and rebuild data structures
        """
        self._swarm_advector.integrate( dt, update_owners=True)
        self.swarm.shadow_particles_fetch()

        self._update_kdtree()
        self._update_surface_normals()

        uw.barrier()

        return

    def all_coords(self):

        """
        Get local and shadow particles
        """
        self.swarm.shadow_particles_fetch()

        dims = self.swarm.particleCoordinates.data.shape[1]

        pc = np.append(self.swarm.particleCoordinates.data,
                       self.swarm.particleCoordinates.data_shadow)

        all_coords = pc.reshape(-1,dims)

        if all_coords.shape[0] == 0:
            return all_coords.T
        else:
            return all_coords



    def compute_interface_proximity(self, coords, distance=None):
        """
        Build a mask of values for points within the influence zone.
        """
        #can be important for parallel
        self.swarm.shadow_particles_fetch()

        #search the Kdtree each side of fault at 0.5*thickness
        if not distance:
            distance = self.thickness*0.5

        if self.empty:
            return np.empty((0,1)), np.empty(0, dtype="int")

        d, p   = self.kdtree.query( coords, distance_upper_bound=distance )

        fpts = np.where( np.isinf(d) == False )[0]

        proximity = np.zeros((coords.shape[0],1))
        proximity[fpts] = self.ID

        return proximity, fpts





    def compute_normals(self, coords, thickness=None):



        # make sure this is called by all procs including those
        # which have an empty self

        self.swarm.shadow_particles_fetch()

        if thickness==None:
            thickness = self.thickness/2.0

        # Nx, Ny = _points_to_normals(self)

        if self.empty:
            return np.empty((0,2)), np.empty(0, dtype="int")

        d, p   = self.kdtree.query( coords, distance_upper_bound=thickness )

        fpts = np.where( np.isinf(d) == False )[0]
        director = np.zeros_like(coords)

        if uw.nProcs() == 1 or self.director.data_shadow.shape[0] == 0:
            fdirector = self.director.data
            #print('1')
        elif self.director.data.shape[0] == 0:
                fdirector = self.director.data_shadow
            #print('2')
        else:
            fdirector = np.concatenate((self.director.data,
                                    self.director.data_shadow))
            #print('3')

        director[fpts] = fdirector[p[fpts]]

        return director, fpts


    def compute_signed_distance(self, coords, distance=None):

        # make sure this is called by all procs including those
        # which have are empty

        #can be important for parallel
        #self.swarm.shadow_particles_fetch()

        #Always need to call self.data on all procs
        all_particle_coords = self.data

        if not distance:
            distance = self.thickness/2.0


        #This hands back an empty array, that has the right shape to be used empty mask
        if self.empty:
            return np.empty((0,1)), np.empty(0, dtype="int")

        #There are a number of cases to consider (probably more compact ways to this)
        #serial is trivial.
        #For parallel, there may be data on local processor, or in shadow zone, or any combination of either.
        # as long as fdirector is the same shape as self.kdtree.data, this will work
        if uw.nProcs() == 1 or self.director.data_shadow.shape[0] == 0:
            fdirector = self.director.data
        elif self.director.data.shape[0] == 0:
            fdirector = self.director.data_shadow
        else:
            #in this case both are non-empty
            fdirector = np.concatenate((self.director.data,
                                    self.director.data_shadow))

        d, p  = self.kdtree.query( coords, distance_upper_bound=distance )

        fpts = np.where( np.isinf(d) == False )[0]

        #this is a bit sneaky, p[fpts] is larger than fdirector: (fdirector[p[fpts]]).shape == vector.shape
        #So this mask is size increasing
        director = fdirector[p[fpts]]
        vector = coords[fpts] - all_particle_coords[p[fpts]]
        #vector = coords[fpts] - self.kdtree.data[p[fpts]]

        signed_distance = np.empty((coords.shape[0],1))
        signed_distance[...] = np.inf

        #row-wise dot product
        sd = np.einsum('ij,ij->i', vector, director)
        signed_distance[fpts,0] = sd[:]
        return signed_distance , fpts


    def _update_surface_normals(self):
        """
        Rebuilds the normals for the string of points
        """

        # This is the case if there are too few points to
        # compute normals so there can be values to remove

        #can be important for parallel
        self.swarm.shadow_particles_fetch()

        #Always need to call self.data on all procs
        all_particle_coords = self.data

        if self.empty:
            self.director.data[...] = 0.0
        else:

            particle_coords = self.swarm.particleCoordinates.data

            #these will hold the normal vector compenents
            Nx = np.empty(self.swarm.particleLocalCount)
            Ny = np.empty(self.swarm.particleLocalCount)

            for i, xy in enumerate(particle_coords):
                r, neighbours = self.kdtree.query(particle_coords[i], k=3)

                # neighbour points are neighbours[1] and neighbours[2]

                XY1 = all_particle_coords[neighbours[1]]
                XY2 = all_particle_coords[neighbours[2]]
                #XY1 = self.kdtree.data[neighbours[1]]
                #XY2 = self.kdtree.data[neighbours[2]]

                dXY = XY2 - XY1

                Nx[i] =  dXY[1]
                Ny[i] = -dXY[0]

                if (self.insidePt):
                    sign = np.sign((self.insidePt[0] - xy[0]) * Nx[i] +
                                   (self.insidePt[1] - xy[1]) * Ny[i])
                    Nx[i] *= sign
                    Ny[i] *= sign


            for i in range(0, self.swarm.particleLocalCount):
                scale = 1.0 / np.sqrt(Nx[i]**2 + Ny[i]**2)
                Nx[i] *= scale
                Ny[i] *= scale


            self.director.data[:,0] = Nx[:]
            self.director.data[:,1] = Ny[:]

        return



    def set_proximity_director(self, swarm, proximityVar, minDistanceFn =fn.misc.constant(1.), maxDistanceFn=fn.misc.constant(1.),
                           locFac=1., searchFac = 2, directorVar=False ):

        #################
        #Part1
        ################

        """
        |  \  / locFacNeg*-1
        |   \/
        |   /\
        |  /  \ locFacPos
         ________
        0-locFac-1
        """

        locFacPos = self.thickness - (self.thickness *locFac)
        locFacNeg = -1.*(self.thickness *locFac)

        #this is a relative thickness, default is 1.
        #try to save an evaluation
        if type(minDistanceFn) == uw.function.misc.constant:
            thickness = minDistanceFn.value
        else:
            thickness= minDistanceFn.evaluate(swarm)

        #First, we want to rebuild the minimum distance...
        sd, pts0 = self.compute_signed_distance(swarm.particleCoordinates.data,
                                                    distance=searchFac*self.thickness)

        #if any Nans appears set them to infs
        sd[np.where(np.isnan(sd))[0]] = np.inf


        #everthing in the min dist halo becomes fault.
        mask = np.logical_and(sd< locFacPos*thickness,           #positive side of fault
                              sd> locFacNeg*thickness)[:,0]      #negative side of fault

        proximityVar.data[mask] = self.ID                      #set to Id

        #################
        #Part2
        ################



        #particles with proximity == self.ID, beyond the retention distance, set to zero
        #I had to do these separately for the two sides of the fault

        #thickness becomes the maxDistanceFunction
        #try to save an evaluation
        if type(maxDistanceFn) == uw.function.misc.constant:
            thickness = maxDistanceFn.value
        else:
            thickness= maxDistanceFn.evaluate(swarm)

        #treat each side of the fault seperately
        #parallel protection
        if sd.shape[0] == proximityVar.data.shape[0]:
            mask1 = operator.and_(sd > locFacPos*thickness, proximityVar.data == self.ID)
            proximityVar.data[mask1] = 0
            mask2 = operator.and_(sd < locFacNeg*thickness,proximityVar.data == self.ID)
            proximityVar.data[mask2] = 0


        #################
        #Part3
        ################


        if directorVar:
            #director domain will be larger than proximity, but proximity will control rheology
            #searchFac *self.thickness should capture the max proximity distance from the fault
            #hence, must be set in relation to the maxDistanceFn
            dv, nzv = self.compute_normals(swarm.particleCoordinates.data, searchFac*self.thickness)
            mask = np.where(proximityVar.data == self.ID)[0]
            directorVar.data[mask, :] = dv[mask, :]







    ## Note that this is strictly 2D  !
    def local_strainrate_fns(self, velocityField, faultNormalVariable, proximityVariable):

        ## This is a quick / short cut way to find the resolved stress components.

        strainRateFn = fn.tensor.symmetric( velocityField.fn_gradient )
        directorVector = faultNormalVariable



        _edotn_SFn = (        directorVector[0]**2 * strainRateFn[0]  +
                        2.0 * directorVector[1]    * strainRateFn[2] * directorVector[0] +
                              directorVector[1]**2 * strainRateFn[1]
                    )


        #initialse the mapping dictionary with zero strain rate for regions of 0 proximity.
        _edotn_SFn_Map    = { 0: 0.0 }
        #for f in self:
        _edotn_SFn_Map[self.ID] =  _edotn_SFn





        _edots_SFn = (  directorVector[0] *  directorVector[1] *(strainRateFn[1] - strainRateFn[0]) +
                        strainRateFn[2] * (directorVector[0]**2 - directorVector[1]**2)
                     )


       #initialse the mapping dictionary with (nearly) zero strain rate for regions of 0 proximity.
        _edots_SFn_Map = { 0: 1.0e-15 }
        _edots_SFn_Map[self.ID] =  _edots_SFn

        edotn_SFn =     fn.branching.map( fn_key = proximityVariable,
                                                     mapping = _edotn_SFn_Map)


        edots_SFn =     fn.branching.map( fn_key = proximityVariable,
                                                     mapping = _edots_SFn_Map )


        return edotn_SFn, edots_SFn





    def get_global_coords(self):
        """
        In some cases we want to expose the global interface line position to all procs
        ***currently not working***
        (Documentation for Allgatherv is pretty sparse.)
        """

        #localData = self.swarm.particleCoordinates.data[:]
        #localData = np.ascontiguousarray(self.swarm.particleCoordinates.data)
        localData = np.copy(self.swarm.particleCoordinates.data)
        localData.astype(dtype=np.float64)

        globalShape = self.swarm.particleGlobalCount
        #print(localData.shape, globalShape)

        sendbuf = localData
        #recvbuf = np.empty([globalShape,2])

        recvbuf =  np.zeros([globalShape,2], dtype='d')
        #comm.Allgatherv([sendbuf, MPI.DOUBLE] , [recvbuf,MPI.DOUBLE])
        comm.Allgatherv(sendbuf, [recvbuf,MPI.DOUBLE])
        return recvbuf


    def neighbourMatrix(self, k= 4, jitter=False):

        """
        neighbourMatrix tries to build neighbour information for an interface2D,
        assuming that the points are unordered.


        For any point, the first neighbour is the closest point.
        The second neighbour is the closest remainng point in the set that forms an angle of more than 90 degree
        to the first neighbour (vector)

        k is the number of neighbours to test before deciding that a nearest neigbour cannot be found (i.e the end of line)

        the information is returned in the form of a dense matrix, where each row corresponds to a point in the interface
        And most rows will have exactly two non-zero element, the indexed of the two nearest neighbour.
        For these points, the matrix is symmetric

        Ideally, there are two rows with only one non-zero column. These are the endpoints.
        (could be better to have a 1 on the diagonal for these?)

        jitter is designed as a way of flushing duplicates, can/should be very small.

        ***This should be safe to run on all processors, and we should always attempt to do so.

        """

        #################
        #Neigbour 1
        #################

        #get the particle coordinates, in the order that the kdTree query naturally returns them
        all_particle_coords = self.data

        if all_particle_coords.shape[1]:
            if jitter:
                #dX = (np.random.rand(self.kdtree.data.shape[0]) - 0.5)*jitter
                dX = (np.random.rand(all_particle_coords.shape[0]) - 0.5)*jitter
                all_particle_coords[:,0] += dX
                all_particle_coords[:,1] += dX

        #All of the functions with barriers should have been called already
        #special cases where self.empty == True
        if len(all_particle_coords) == 1:
            a = np.array([1])
            a2 = np.expand_dims(a,axis=1)
            return a2
        elif len(all_particle_coords) == 2:
            return np.fliplr(np.eye(2))


        queryOut = self.kdtree.query(all_particle_coords, k=all_particle_coords.shape[0] )
        ids = queryOut[1]
        #build the matrix of neighbour -adjacency
        AN = np.zeros((all_particle_coords.shape[0],all_particle_coords.shape[0] ))

        #First add the nearest neighbour, which is column one in ids (ids[np.arange(len(AN)), 1])
        AN[ids[:,0],ids[np.arange(len(AN)), 1]] =  1

        #################
        #Neigbour 2
        #################
        coords = all_particle_coords[ids]

        #for each row in vectorArray, every col. is the vector between the neighbours (distance ordered) and the reference particle
        #None the None arg needed to get the broadcasting right
        vectorArray = ( all_particle_coords[ids[:,:]] - all_particle_coords[ids[:,None,0]])

        #this computes the dot product of the neighbour 1 vector with all other neighbour vectors
        dotProdCompare = np.einsum('ijk, ik->ij', vectorArray[:,:,:], vectorArray[:,1,:])
        dotProdCompare[:,1]

        #find the first point for which the position vector has a negative dot product with the nearest neighbour

        negDots = dotProdCompare < 0.

        #Here's where we limit the search to k nearest neighbours
        if k:
            negDots[:,k:] = False

        #cols holds the index the column of the first negative entry (negative dot-product).
        cols = np.argmax(negDots[:,:], axis = 1)
        #Note if cols is zero, it means no obtuse neighbour was found - likely an end particle.
        answer = ids[np.arange(all_particle_coords.shape[0]),cols]
        #now add the first subsequent neighbour that is obtuse to the first
        AN[ids[:,0],answer] =  1
        #Now remove diagonals - these were any particles where a nearest obtuse neighbour couldn't be found
        np.fill_diagonal(AN, 0)

        return AN


    def laplacianMatrix(self, k = 4):
        """
        """

        #dims = self.kdtree.data.shape[1]
        dims = self.data.shape[1]

        #Get neighbours
        #all_particle_coords = self.kdtree.data
        all_particle_coords = self.data

        #if len(all_particle_coords) == 1: #safeguard for little arrays
        #    return np.array([0.])

        #create 2s on the diagonal
        L = 2.*np.eye(all_particle_coords.shape[0])

        A = self.neighbourMatrix(k= k)

        #All of the functions / methods with basrriers need to go above this line
        if len(all_particle_coords) == 1: #safeguard for little arrays
            return np.array([0.])

        #set all neighbours to -1
        L[A == 1] = -1
        #Find rows that only have one neighbour (these are probably/hopefully endpoints)
        mask = np.where(A.sum(axis=1) == 1)
        #Set those rows to zero
        L[mask,:]  = 0
        #And set the diagonal back to 2. (The Laplacian operator should just return the particle position)
        #L[mask,mask] = 2

        return 0.5*L #right?


    def pairDistanceMatrix(self):
        """
        """
        #partx = self.kdtree.data[:,0]
        #party = self.kdtree.data[:,1]

        all_particle_coords = self.data

        if all_particle_coords.shape[1]:


            partx = all_particle_coords[:,0]
            party = all_particle_coords[:,1]


            dx = np.subtract.outer(partx , partx )
            dy = np.subtract.outer(party, party)
            distanceMatrix = np.hypot(dx, dy)

        else:
            distanceMatrix = np.empty(0)


        return distanceMatrix


class interface_collection(list):
    '''
    Collection (list) of interface2D objects which together define the global rheology
    '''
    def __init__(self, interface_list=None):

        #initialise parent Class
        super(interface_collection, self).__init__()

        if interface_list != None:
            for interface in interface_list:
                if isinstance(interface, interface2D):
                    super(interface_collection, self).append(interface)
                else:
                    print "Non interface object ", interface, " not added to collection"

        return

    def append(self, interface):
        '''

        '''
        if isinstance(interface, interface2D): #
            super(interface_collection, self).append(interface)
        else:
            print "Non interface object ", line, " not added to collection"


    ## Note that this is strictly 2D  !
    def global_line_strainrate_fns(self, velocityField, lineNormalVariable, proximityVariable):

        ## This is a quick / short cut way to find the resolved stress components.

        strainRateFn = fn.tensor.symmetric( velocityField.fn_gradient )
        directorVector = lineNormalVariable



        _edotn_SFn = (        directorVector[0]**2 * strainRateFn[0]  +
                        2.0 * directorVector[1]    * strainRateFn[2] * directorVector[0] +
                              directorVector[1]**2 * strainRateFn[1]
                    )

        # any non-zero proximity requires the computation of the above

        _edotn_SFn_Map    = { 0: 0.0 }
        for f in self:
            _edotn_SFn_Map[f.ID] =  _edotn_SFn





        _edots_SFn = (  directorVector[0] *  directorVector[1] *(strainRateFn[1] - strainRateFn[0]) +
                        strainRateFn[2] * (directorVector[0]**2 - directorVector[1]**2)
                     )


        _edots_SFn_Map = { 0: 1.0e-15 }

        for f in self:
            _edots_SFn_Map[f.ID] =  _edots_SFn


        edotn_SFn =     fn.branching.map( fn_key = proximityVariable,
                                                     mapping = _edotn_SFn_Map)


        edots_SFn =     fn.branching.map( fn_key = proximityVariable,
                                                     mapping = _edots_SFn_Map )


        return edotn_SFn, edots_SFn
