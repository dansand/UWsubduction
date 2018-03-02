
import numpy as np
#import underworld as uw
from mpi4py import MPI
comm = MPI.COMM_WORLD

from scipy.spatial import cKDTree as kdTree

class globalLine2D(object):
    """
    Should provide similar func. to Louis' original code, but ignores swarms.
    Designed for cases where we're not advecting the marker line,
    and the parllel nature of the lines is burdensome
    """


    def __init__(self, mesh, velocityField, pointsX, pointsY, fthickness, fID, insidePt=(0.0,0.0)):

        self.empty = False

        # Should do some checking first

        self.mesh = mesh
        self.velocity = velocityField
        self.thickness = fthickness
        self.ID = fID
        self.insidePt = insidePt
        self.director = None


        self.director = np.zeros((pointsX.shape[0], 2))

        self.particleCoordinates = np.column_stack((pointsX, pointsY))

        self._update_kdtree()
        self._update_surface_normals()

        return



    def add_points(self, pointsX, pointsY):
        pass

    def rebuild(self):

        self._update_kdtree()
        self._update_surface_normals()

        return


    def _update_kdtree(self):

        self.empty = False

        all_particle_coords = self.particleCoordinates

        if len(all_particle_coords) < 3:
            self.empty = True

            self.kdtree = kdTree(np.empty((2,0)))
        else:
            self.kdtree = kdTree(all_particle_coords)

        return


    def advection(self, dt):
        """
        Update marker swarm particles as material points and rebuild data structures
        """
        pass





    def compute_marker_proximity(self, coords, distance=None):
        """
        Build a mask of values for points within the influence zone.
        """


        if not distance:
            distance = self.thickness


        d, p   = self.kdtree.query( coords, distance_upper_bound=distance )

        fpts = np.where( np.isinf(d) == False )[0]

        proximity = np.zeros((coords.shape[0],1))
        proximity[fpts] = self.ID

        return proximity, fpts





    def compute_normals(self, coords, thickness=None):



        if thickness==None:
            thickness = self.thickness

        # Nx, Ny = _points_to_normals(self)


        d, p   = self.kdtree.query( coords, distance_upper_bound=thickness )

        fpts = np.where( np.isinf(d) == False )[0]
        director = np.zeros_like(coords)

        #the fault director
        fdirector = self.director

        director[fpts] = fdirector[p[fpts]]

        return director, fpts


    def compute_signed_distance(self, coords, distance=None):

        # make sure this is called by all procs including those
        # which have an empty self

        if not distance:
            distance = self.thickness

        #print(self.director.data.shape, self.director.data_shadow.shape)

        #the fault director
        fdirector = self.director

        d, p  = self.kdtree.query( coords, distance_upper_bound=distance )

        fpts = np.where( np.isinf(d) == False )[0]

        director = np.zeros_like(coords)  # Let it be zero outside the region of interest
        director = fdirector[p[fpts]]

        #print('dir. min', np.linalg.norm(director, axis = 1).min())

        vector = coords[fpts] - self.kdtree.data[p[fpts]]


        dist = np.linalg.norm(vector, axis = 1)

        signed_distance = np.empty((coords.shape[0],1))
        signed_distance[...] = np.inf

        sd = np.einsum('ij,ij->i', vector, director)
        signed_distance[fpts,0] = sd[:]
        #signed_distance[:,0] = d[...]

        #return signed_distance, fpts
        #signed_distance[fpts,0] = dist[:]
        return signed_distance , fpts


    def _update_surface_normals(self):
        """
        Rebuilds the normals for the string of points
        """

        # This is the case if there are too few points to
        # compute normals so there can be values to remove

        if self.empty:
            self.director[...] = 0.0
        else:

            particle_coords = self.particleCoordinates

            #these will hold the normal vector compenents
            Nx = np.empty(self.particleCoordinates.shape[0])
            Ny = np.empty(self.particleCoordinates.shape[0])

            for i, xy in enumerate(particle_coords):
                r, neighbours = self.kdtree.query(particle_coords[i], k=3)

                # neighbour points are neighbours[1] and neighbours[2]

                XY1 = self.kdtree.data[neighbours[1]]
                XY2 = self.kdtree.data[neighbours[2]]

                dXY = XY2 - XY1

                Nx[i] =  dXY[1]
                Ny[i] = -dXY[0]

                if (self.insidePt):
                    sign = np.sign((self.insidePt[0] - xy[0]) * Nx[i] +
                                   (self.insidePt[1] - xy[1]) * Ny[i])
                    Nx[i] *= sign
                    Ny[i] *= sign


            for i in range(0, self.particleCoordinates.shape[0]):
                scale = 1.0 / np.sqrt(Nx[i]**2 + Ny[i]**2)
                Nx[i] *= scale
                Ny[i] *= scale


            self.director[:,0] = Nx[:]
            self.director[:,1] = Ny[:]

        return
