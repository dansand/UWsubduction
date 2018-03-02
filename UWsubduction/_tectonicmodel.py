import numpy as np
import underworld as uw
from underworld import function as fn
import glucifer
import networkx as nx
import operator



class TectonicModel(nx.DiGraph):

    """

    ***note on has_edge()***
    I intend most methods requiring two plates to take a tuple (plate1Id, plate2Id)
    the DiGraph has_edge() method can be handed a tuple, if it prececeeded by a *
    .has_edge(*(plate1Id, plate2Id)) //or// .has_edge(plate1Id, plate2Id)



    Formatting:
    use underscores for methods

    """


    def __init__(self, mesh, starttime, endtime, dt):

        ########Trying various ways to init the parent class
        #super(nx.DiGraph, self).__init__(*args)
        #super().__init__(*args)
        nx.DiGraph.__init__(self)
        ################################

        self.times = np.arange(starttime, endtime, dt)
        #self.add_node('times', times=self.times)
        self.plateIdUsedList = []
        self.plateIdDefaultList = list(np.arange(1, 101))
        self.plateDummyId = -99

        #mesh and coordinate functions
        self.mesh = mesh
        self._coordinate = fn.input()
        self._xFn = self._coordinate[0]


    #################################
    ##Some utility Fns
    #################################

    @staticmethod
    def b2f(fN):

        #create map dict
        b2f_ = {}
        b2f_[True] = 1.
        b2f_[False] = 0.

        fnConv = fn.branching.map(fn_key = fN,
                              mapping = b2f_ )
        return fnConv

    @staticmethod
    def t2f(fN):


        fnConv = fn.branching.conditional( ((fN, False ),
                                           (True,                      True )  ))

        return fnConv


    @staticmethod
    def f2b(fN):
        #not totally safe!
        #Underlying arrays don't support equality comparison
        fnConv = fn.branching.conditional( ((fN > 0.999, True ),
                                            (fN < 1e-3, False ),
                                           (True,                      False )  ))

        return fnConv

    #############################


    #using getters as I'm unsure how this part of the code may evolve,
    #i.e. the coupling between underworld objects and the current Class

    @property
    def xFn(self):
        return self._xFn

    @property
    def minX(self):
        return self.mesh.minCoord[0]

    @property
    def maxX(self):
        return self.mesh.maxCoord[0]

    @property
    def undirected(self):
        #%#%#%#%#%#%#%#%#%#%#%#%#%#%
        #need to make sure this updates. It should
        #%#%#%#%#%#%#%#%#%#%#%#%#%#%
        return self.to_undirected()







    #################################
    #Read from Dict function to allow checkpointing
    #################################
    def pop_from_dict_of_lists(self, d):

        """Return a graph from a dictionary of lists.
        Adapted from the networkX function

        """

        self.add_nodes_from(d)
        self.add_edges_from(((u, v, data)
                              for u, nbrs in d.items()
                              for v, data in nbrs.items()))


    #################################
    ##General graph query / utilities
    #################################

    def get_index_at_time(self, time):

        """return an index for the velocity/time array, using closest value"""
        ix_ = np.argmin(np.abs((self.times - time)))
        return ix_

    def plate_velocity(self, plateId, time=False):
        """get the velocity array of a given plate, or velcity value if time provided """
        if time != False:
            ix_ = self.get_index_at_time(time)
            return self.node[plateId]['velocities'][ix_]
        else:
            return self.node[plateId]['velocities']

    def bound_velocity(self, platePair, time=False):
        """get the velocity array of a given boundary, or velcity value if time provided """
        if time is not False:
            ix_ = self.get_index_at_time(time)
            return self.get_edge_data(*platePair)['velocities'][ix_]
        else:
            return self.get_edge_data(*platePair)['velocities']

    def upper_plate_vel(self, platePair, time=False):
        """get the upper plate velocity for a subduction boundary, or velcity value if time provided """
        assert self.is_subduction_boundary(platePair), 'not a subduction boundary'
        sp = self.subduction_edge_order(platePair)[1]
        if time is not False:
            ix_ = self.get_index_at_time(time)
            return self.plate_velocity(plateId)[ix_]
        else:
            return self.plate_velocity(plateId)

    def plate_has_vel(self, plateId, time):
        r = False
        if not np.isnan(self.plate_velocity(plateId, time=time) ):
            r = True
        return r

    def bound_has_vel(self, platePair, time):
        r = False
        if not np.isnan(self.bound_velocity(platePair, time=time) ):
            r = True
        return r

    def upper_plate_has_vel(self, platePair, time):
        up = self.subduction_edge_order(platePair)[1]
        r = False
        if self.plate_has_vel(up, time):
            r = True
        return r


    def connected_plates(self, plateId):
        """return the connected plates for a given plate"""

        #return (list(set([x for x in nx.all_neighbors(self, plateId)])))
        return self.undirected.neighbors(plateId)


    def is_subduction_boundary(self, platePair):
        """return a boolean if a plate boundary is a subduction zone"""
        result = False

        #possible options are: no connection between nodes (False, False)
        #           :  two way connecrion (ridge) (True, True)
        #           : one way connection (Truem False) ....

        if self.has_edge(*(platePair[0], platePair[1])) != self.has_edge(*(platePair[1], platePair[0])):
            result = True
        return result

    #this should be made safe for the directedness...right?
    def get_bound_loc(self, platePair):
        """return the location (x coord) of a plate boundary"""
        return self.get_edge_data(*platePair)['loc']

    def set_bound_loc(self, platePair, newloc):
        """change the location (x coord) of a plate boundary"""
        self.get_edge_data(*platePair)['loc'] = newloc

        #if the boundary is a ridge, we need to update the edge in the other direction
        if self.is_ridge(platePair):
            otherDirection = (platePair[1], platePair[0])
            self.get_edge_data(*otherDirection)['loc'] = newloc


    def is_ridge(self, platePair):
        """return a boolean if a plate boundary is a ridge"""
        result = False
        if self.has_edge(*(platePair[0], platePair[1])) and self.has_edge(*(platePair[0], platePair[1])) == self.has_edge(*(platePair[1], platePair[0])):
            result = True
        return result

    def is_self_loop(self, platePair):
        """return a boolean if a plate boundary is a self loop (domain boundary / ficticious boundary)"""
        pass

    def subduction_edge_order(self, platePair):
        """given a subduction boundary (platePair) return the plateIds with the subdction plate first"""

        if self.has_edge(*(platePair[0], platePair[1])) and not self.has_edge(*(platePair[1], platePair[0])):
            return [platePair[0], platePair[1]]
        elif self.has_edge(*(platePair[1], platePair[0])) and not self.has_edge(*(platePair[0], platePair[1])):
            return [platePair[1], platePair[0]]
        else:
            raise ValueError("boundary does not exist, or not a subduction boundary")

    def subduction_direction(self, platePair):
        """return the unit vector (c-comp) pointing from te subducting plate to the upper plate"""
        if self.is_subduction_boundary((platePair[0], platePair[1])):

            segde = self.subduction_edge_order((platePair[0], platePair[1]))
            #sz goes form segde[0] to segde[1]

            if  np.sort(self.get_boundaries(segde[0])).mean() <  np.sort(self.get_boundaries(segde[1])).mean():
                return 1.
            else:
                return -1.
        else:
            raise ValueError("boundary does not exist, or not a subduction boundary")



    #currently not safe for multiple subduction zones
    def subduction_boundary_from_plate(self, plateId):
        "given a plate, determine the subduction boundary as a platePair"

        if self.is_subducting_plate(plateId):

            cps = self.connected_plates(plateId )
            if self.has_edge(*(plateId, cps[0])) and self.has_edge(*(cps[0], plateId)):
                return self.subduction_edge_order((cps[1], plateId))
            else:
                return self.subduction_edge_order((cps[0], plateId))
        else:
            raise ValueError("not a subduction boundary")




    def is_subducting_plate(self, plateId):
        """return a boolean if a give plate has a subduction zone"""

        sp = False
        cps = self.connected_plates(plateId )

        for b in cps:
            #this checks that the edge is a ridge (both directions)
            if self.has_edge(*(plateId, b)) and self.has_edge(*(b, plateId)) and sp == False:
                sp = False
            else:
                #this checks the direction the subduction zone
                if self.has_edge(*(plateId, b)):
                    sp = True

        return sp



    #def has_boundary_plate(self, plateId):
    #    return  plateId in self[plateId].keys() and len(self.undirected[plateId].keys()) ==2

    #rename to get_plate_boundary_locs?
    def get_boundaries(self, plateId):

        """return boundary locations for a given plate"""

        if len(self.connected_plates(plateId)) == 2:


            cps = self.connected_plates(plateId)
            #cps = self.undirected.neighbors(plateId)
            #print(cps)
            loc1 = self.undirected[plateId][cps[0]]['loc']
            loc2 = self.undirected[plateId][cps[1]]['loc']
            return [loc1, loc2]

        else:
            print('plate does not have 2 boundaries. Cannot define extent.')
            return []

    #################################
    ##Adding plates / plate boundaries
    #################################
    """add a plate (node) to the the tectModel (graph)"""

    def add_plate(self, ID = False, velocities = False):
        #default is an array of nans
        if velocities is False:      #use is to test for identity
            vels = np.empty(len(self.times))
            vels.fill(np.nan)
        elif type(velocities) == int or type(velocities) == float:
            vels = np.ones(len(self.times ))*velocities
        elif len(velocities) == len(self.times ):
            vels = velocities
        elif len(velocities) != len(self.times ):
            raise ValueError("velocities must be a single float/int or list/array of length self.times ")

        #i'm using -99 as a dummy value (fall back) in the plateIDFn
        #having a plate Id of the same value would cause much confusion

        #it's generaly good to keep 0 as the null value when mapping proximity,
        #so we also reserve 0

        if ID == self.plateDummyId or ID == 0:
            raise ValueError("plate ID reserved, choose another ID")

        if not ID:
            ID = self.plateIdDefaultList[0]

        if ID not in self.plateIdUsedList:
            self.add_node(ID, velocities= vels)
            self.plateIdUsedList.append(ID)
            self.plateIdDefaultList.remove(ID)
        else:
            raise ValueError("plate ID already assigned")

    def add_subzone(self, subPlate, upperPlate, loc, subInitAge=0.0, upperInitAge=0.0, velocities = False):

        """add a subduction boundary (directed edge) to the the tectModel (graph)
          a single directed egde is added to the graph
          subPlate => upperPlate
          the subduction zone will point/dip from the subPlate to the upperPlate
        """

        #default is an array of nans
        if velocities == False:
            vels = np.empty(len(self.times))
            vels.fill(np.nan)
        elif type(velocities) == int or type(velocities) == float:
            vels = np.ones(len(self.times ))*velocities
        elif len(velocities) == len(self.times ):
            vels = velocities
        elif len(velocities) != len(self.times ):
            raise ValueError("velocities must be a single float/int or list/array of length self.times ")

        #check whether the plate boundary can be simply inserted
        if len(self.connected_plates(subPlate)) <=2 and len(self.connected_plates(upperPlate)) <=2 :

            self.add_edge(subPlate, upperPlate,
                          loc= loc,
                          ages = {subPlate:subInitAge,
                                  upperPlate:upperInitAge},
                          velocities= vels)
        else:
            print('plate already has 2 boundaries. Wait for plate transfer to be implemented')

    def add_ridge(self, plate1, plate2, loc, plate1InitAge=0.0, plate2InitAge=0.0, velocities = False):

        """add a ridge boundary (edge) to the the tectModel (graph)
           to denote a ridge the directed graph has a two edges:
           plate1 => plate2 & plate2 => plate1
        """

        #default is an array of nans
        if velocities == False:
            vels = np.empty(len(self.times))
            vels.fill(np.nan)
        elif type(velocities) == int or type(velocities) == float:
            vels = np.ones(len(self.times ))*velocities
        elif len(velocities) == len(self.times ):
            vels = velocities
        elif len(velocities) != len(self.times ):
            raise ValueError("velocities must be a single float/int or list/array of length self.times ")

        #check whether the plate boundary can be simply inserted
        if len(self.connected_plates(plate1)) <=2 and len(self.connected_plates(plate2)) <=2:

            #note that if plate1 == plate2, there will only be one entry in teh age dictionary

            self.add_edge(plate1, plate2, loc= loc,
                          ages = {plate1:plate1InitAge,
                                  plate2:plate2InitAge},
                          velocities= vels)
            self.add_edge(plate2, plate1, loc= loc, ages = {plate1:plate1InitAge,
                                  plate2:plate2InitAge},
                          velocities= vels)
        else:
            print('plate already has 2 boundaries. Wait for plate transfer to be implemented')

    def add_left_boundary(self, plate,  plateInitAge=0.0, velocities = False):


        """add a left domain boundary / ficticious plate bounary to the graph
           to denote a domain boundary we add a 'self loop' edge:
           plate => plate
        """

        #default is an array of nans
        if velocities == False:
            vels = np.empty(len(self.times))
            vels.fill(np.nan)
        elif type(velocities) == int or type(velocities) == float:
            vels = np.ones(len(self.times ))*velocities
        elif len(velocities) == len(self.times ):
            vels = velocities
        elif len(velocities) != len(self.times ):
            raise ValueError("velocities must be a single float/int or list/array of length self.times ")

        #check whether the plate boundary can be simply inserted
        if len(self.connected_plates(plate)) <=2:



            #note that if plate1 == plate2, there will only be one entry in teh age dictionary

            self.add_edge(plate, plate, loc= self.minX,
                          ages = {plate:plateInitAge},
                          velocities= vels)
        else:
            print('plate already has 2 boundaries. Wait for plate transfer to be implemented')

    def add_right_boundary(self, plate,  plateInitAge=0.0, velocities = False):

        """add a right domain boundary / ficticious plate bounary to the graph
           to denote a domain boundary we add a 'self loop' edge:
           plate => plate
        """
        #default is an array of nans
        if velocities == False:
            vels = np.empty(len(self.times))
            vels.fill(np.nan)
        elif type(velocities) == int or type(velocities) == float:
            vels = np.ones(len(self.times ))*velocities
        elif len(velocities) == len(self.times ):
            vels = velocities
        elif len(velocities) != len(self.times ):
            raise ValueError("velocities must be a single float/int or list/array of length self.times ")
        #check whether the plate boundary can be simply inserted
        if len(self.connected_plates(plate)) <=2:

            #note that if plate1 == plate2, there will only be one entry in teh age dictionary

            self.add_edge(plate, plate, loc= self.maxX,
                          ages = {plate:plateInitAge},
                          velocities= vels)
        else:
            print('plate already has 2 boundaries. Wait for plate transfer to be implemented')


    #################################
    ##Mask Functions...
    #################################

    #mask functions return boolean by default,
    #but can return floating point with the argument out = 'num'

    def plate_boundary_mask_fn(self, dist, bound=False, out = 'bool'):
        """build an underworld boolean function that is False in the neigbourhood (+/- dist)
        of plate boundaries and True everywhere else.
        Applies to all plate boudaries if bound=False, or boundary if bound is a plate pair (id1, id2)
        """

        condList = []
        if not bound: #compute mask for all boundaries
            for e in self.undirected.edges():
                loc = self.undirected.get_edge_data(*e)['loc']

                cond = fn.math.abs(loc - self.xFn) <= dist
                condList.append((cond, True))


        else:
                loc = self.undirected.get_edge_data(*bound)['loc']
                cond = fn.math.abs(loc - self.xFn) <= dist
                condList.append((cond, True))


        condList.append((True, False))
        boundMaskFn = fn.branching.conditional( condList)

        if out == 'bool':
            return boundMaskFn
        elif out == 'num':
            return self.b2f(boundMaskFn)
        else:
            raise ValueError("out must be one of 'bool or 'num'")

    def ridge_mask_fn(self, dist, out = 'bool'):

        """build an underworld boolean function that is False in the neigbourhood (+/- dist)
        of all Ridge boundaries and True everywhere else"""
        condList = []
        for e in self.undirected.edges():
            if not self.is_subduction_boundary(e):
                loc = self.undirected.get_edge_data(*e)['loc']


                cond = fn.math.abs(loc - self.xFn) <= dist
                condList.append((cond, True))

        condList.append((True, False))
        ridgeMaskFn = fn.branching.conditional( condList)
        if out == 'bool':
            return ridgeMaskFn
        elif out == 'num':
            return self.b2f(ridgeMaskFn)
        else:
            raise ValueError("out must be one of 'bool or 'num'")

    #This is currently symmetric around the SZ loc.  May want to change this so it only masks the subducting plate
    def subduction_mask_fn(self, dist, out = 'bool'):
        """
        build an underworld boolean function that is False in the neigbourhood (+/- dist)
        of all Subduction boundaries and True everywhere else
        """


        condList = []
        for e in self.undirected.edges():
            if self.is_subduction_boundary(e):
                loc = self.undirected.get_edge_data(*e)['loc']


                cond = fn.math.abs(loc - self.xFn) <= dist
                condList.append((cond, True))

        condList.append((True, False))
        subMaskFn = fn.branching.conditional( condList)

        if out == 'bool':
            return subMaskFn
        elif out == 'num':
            return self.b2f(subMaskFn)
        else:
            raise ValueError("out must be one of 'bool or 'num'")


    def plate_interior_mask_fn(self, relativeWidth = 1.0, minPlateLength = 0., plate=False, out = 'bool', invert=False):

        """
        relativeWidth is the size of the Mask width (True) relative to the plate width
        relativeWidth should be between 0 - 1
        """

        condList = []

        if not plate: #compute mask for all plates
            for plateId in self.nodes():
                bounds = np.sort(self.get_boundaries(plateId))
                midpoint = bounds.mean()
                #print(bounds)
                length = max(0., bounds[1] - bounds[0])*relativeWidth
                if length < minPlateLength:
                    pass
                else:
                    cond = fn.math.abs(self.xFn - midpoint) <= 0.5*length
                    if invert == True:
                        condList.append((cond, False))
                    else:
                        condList.append((cond, True))

        else: #compute for single plates
            bounds = np.sort(self.get_boundaries(plate))
            midpoint = bounds.mean()
            #print(bounds)
            length = max(0., bounds[1] - bounds[0])*relativeWidth
            if length < minPlateLength:
                pass
            else:
                cond = fn.math.abs(self.xFn - midpoint) <= 0.5*length
                if invert == True:
                    condList.append((cond, False))
                else:
                    condList.append((cond, True))

        if invert == True:
            condList.append((True, True))
        else:
            condList.append((True, False))

        intMaskFn = fn.branching.conditional( condList)

        if out == 'bool':
            return intMaskFn
        elif out == 'num':
            return self.b2f(intMaskFn)
        else:
            raise ValueError("out must be one of 'bool or 'num'")

    def variable_boundary_mask_fn(self, distMax, distMin=0., relativeWidth = 1.0,
                                  minPlateLength = 0., bound=False, out = 'bool', boundtypes='all'):

        """
        relativeWidth is the size of the Mask width (True) relative to the plate width
        """

        condList = []

        if not bound: #compute mask for all boundaries
            for e in self.undirected.edges():
                if boundtypes == 'ridge' and not self.is_ridge(e):
                    continue #break current cycle of loop
                if boundtypes == 'sub' and not self.is_subduction_boundary(e):
                    continue
                boundloc = self.undirected.get_edge_data(*e)['loc']
                p1 = e[0]
                b1 = self.get_boundaries(p1)
                p2 = e[1]
                b2 = self.get_boundaries(p2)

                #Define a right and left plate with reference to the plate boundary
                if max(b2) > max(b1):
                    rp = p2
                    rb = b2
                    lp = p1
                    lb = b1
                else:
                    rp = p1
                    rb = b1
                    lp = p2
                    lb = b2


                if (rb[1] - rb[0]) < minPlateLength:
                    rightDist = 0.
                else:
                    rightDist = max( min((rb[1] - rb[0])*relativeWidth, distMax), distMin)
                if (lb[1] - lb[0]) < minPlateLength:
                    leftDist = 0.
                else:
                    leftDist = max( min((lb[1] - lb[0])*relativeWidth, distMax), distMin)

                #print(leftDist, rightDist)
                cond = operator.and_ ((self.xFn - boundloc) <= rightDist, (self.xFn - boundloc) >= -1.*leftDist)
                condList.append((cond, True))


        else: #compute for single boundary

            boundloc = self.undirected.get_edge_data(*bound)['loc']
            p1 = bound[0]
            b1 = self.get_boundaries(p1)
            p2 = bound[1]
            b2 = self.get_boundaries(p2)

            #Define a right and left plate with reference to the plate boundary
            if max(b2) > max(b1):
                rp = p2
                rb = b2
                lp = p1
                lb = b1
            else:
                rp = p1
                rb = b1
                lp = p2
                lb = b2

            if (rb[1] - rb[0]) < minPlateLength:
                rightDist = 0.
            else:
                rightDist = max( min((rb[1] - rb[0])*relativeWidth, distMax), distMin)
            if (lb[1] - lb[0]) < minPlateLength:
                leftDist = 0.
            else:
                leftDist = max( min((lb[1] - lb[0])*relativeWidth, distMax), distMin)

            #print(leftDist, rightDist)
            cond = operator.and_ ((self.xFn - boundloc) <= rightDist, (self.xFn - boundloc) >= -1.*leftDist)
            condList.append((cond, True))

        #add the fallback condition
        condList.append((True, False))
        intMaskFn = fn.branching.conditional( condList)

        if out == 'bool':
            return intMaskFn
        elif out == 'num':
            return self.b2f(intMaskFn)
        else:
            raise ValueError("out must be one of 'bool or 'num'")




    def combine_mask_fn(self, mask1, mask2):

        """add Boolean mask functions"""

        #We can `add' mask functions using the operator package
        combMaskFn = operator.or_(mask1, mask2)
        return combMaskFn

    def interior_mask_fn_dict(self, **kwargs):

        """
        Want to be able to return any of the mask functions as a dictionary:
        id:maskFn
        """
        d = {}
        for plateId in self.nodes():
            mfn = self.plate_interior_mask_fn(plate=plateId, **kwargs)
            d[plateId ] = mfn

        return d


    #################################
    ##Other fuctions, plateId, Age, nodes, etc
    #################################


    def plate_id_fn(self, boundtol=1e-5, maskFn = fn.misc.constant(False)):

        """Return an Underworld function that maps the plate domain (in the xcoord) to the plateId
           The resulting function is simpley a set of n integer values in x: F(x) => Zn
           If plates do not cover the entire domain the special Index self.plateDummyId will appear
        """

        condList = []
        condList.append((maskFn, self.plateDummyId))
        for n in self.nodes():
            bounds  = np.sort(self.get_boundaries(n))

            #edgetol = 1e-4
            if fn.math.abs(bounds[0] - self.minX) < boundtol:
                lb = bounds[0] - boundtol
            else:
                lb = bounds[0]

            if fn.math.abs(bounds[1] - self.maxX) > boundtol:
                ub = bounds[1] + boundtol
            else:
                ub = bounds[1]

            cond = operator.and_(self.xFn >= lb, self.xFn < ub)
            condList.append((cond, n))

        #maskFnCond = False


        condList.append((True, self.plateDummyId))

        idFn = fn.branching.conditional( condList)
        return idFn


    def plate_vel_node_fn(self, time, boundtol=1e-5, maskFn = fn.misc.constant(True)):

        """
        This method is very similar to the plateIdFn but here we mask for any plate where viscosity is not set.
        The resulting Underworld function can be used to set velocities on nodes local to the plate
        """

        #we need to flip the mask function before handing to plate_id_fn
        #as trues go to falses in the line: condList.append((maskFn, self.plateDummyId))
        maskFn_ = self.t2f(maskFn)

        condList = []
        condList.append((maskFn_, self.plateDummyId))
        for n in self.nodes():


            bounds  = np.sort(self.get_boundaries(n))

            #edgetol = 1e-4
            if fn.math.abs(bounds[0] - self.minX) < boundtol:
                lb = bounds[0] - boundtol
            else:
                lb = bounds[0]

            if fn.math.abs(bounds[1] - self.maxX) > boundtol:
                ub = bounds[1] + boundtol
            else:
                ub = bounds[1]

            cond = operator.and_(self.xFn >= lb, self.xFn < ub)


            idx_ = [self.get_index_at_time(time)]
            v = self.plate_velocity(n)[idx_]

            if np.isnan(v):
                n = self.plateDummyId

            condList.append((cond, n))


        condList.append((True, self.plateDummyId))

        idFn = fn.branching.conditional( condList)

        #now call a separate funtion to get the nodes
        nodes = self.get_vel_nodes( idFn)


        return nodes


    def plate_age_fn(self):
        """
        Returns a dictionary of Undeworld functions (F(x)) representing the plate age, based on linear interpolation between
        ages provided at boundary locations.

        The dictionary can be used in conjuction with the plate ID function (plate_id_fn())
        to produce the piecewise plate age

        Beacuse, this is a dictionary of functions,  individual items (functions) can be altered / switched out
        To create more complex starting configurations.



        """
        ageFnDict = {0:fn.misc.constant(0.)}

        #As in many cases, we iterate through the undirected version of the graph, which is simpler
        uG = self.undirected

        for n in uG.nodes():
            ns = uG.neighbors(n)
            locAge1 = (uG[n][ns[0]]['loc'],  uG[n][ns[0]]['ages'][n])
            locAge2 = (uG[n][ns[1]]['loc'],  uG[n][ns[1]]['ages'][n])

            #Age gradient
            Agrad = (locAge2[1] - locAge1[1])/(locAge2[0] - locAge1[0])

            ageFn=  locAge1[1] + Agrad*(self.xFn - locAge1[0])
            ageFnDict[n] = ageFn

        return ageFnDict


    def get_vel_nodes(self, nodeIdFn):
        """given a nodeIdFn (i.e plate_vel_node_fn()) return a complet set of local surface nodes.
        these are typically used as the nodes in a Stokes Dirichlet condition

        *Nodes here refers to the nodes of the FEM mesh, not the TectModel graph.
        """
        if self.mesh.specialSets['MaxJ_VertexSet'].data.shape[0]:
            mask =  (nodeIdFn.evaluate(self.mesh.specialSets['MaxJ_VertexSet']) != self.plateDummyId)[:,0]
            nodes = self.mesh.specialSets['MaxJ_VertexSet'].data[mask]
        else:
            nodes = np.empty(0)

        #mask = (nodeIdFn.evaluate(self.mesh.specialSets['MaxJ_VertexSet']) != self.plateDummyId)[:,0]
        #nodes = self.mesh.specialSets['MaxJ_VertexSet'].data[mask]
        return nodes


    def plateVelFn(self, time, pIdFn, scaleFn = fn.misc.constant(1)):

        """return and underworld fn.branching.map linking plateIds (integers) to the specified plate velocities
        at a given time. The map is used to (re)set the Dirichlet condition on the surface"""

        velFnDict = {99:0.}
        uG = self.undirected

        for n in self.nodes():
            idx_ = [self.get_index_at_time(time)]
            v = self.plate_velocity(n)[idx_]
            velFnDict[n] = v[0]

        #fnVel_map = fn.branching.map(fn_key = pIdFn ,
        #                  mapping = velFnDict)

        #return velFnDict

        fnVel_map = fn.branching.map(fn_key = pIdFn ,
                             mapping = velFnDict)

        return fnVel_map


    def subZoneAbsDistFn(self, bigNum = 1e3, upper = False):

        """
        This is currently not valid for multiple subduction zones attached to one plate
        """
        #subZoneAbsDistDict = {0:fn.misc.constant(bigNum)}

        subZoneAbsDistDict = {}
        uG = self.undirected
        if upper is False:
            for n in uG.nodes():
                if not self.is_subducting_plate(n):
                    subZoneAbsDistDict[n] = fn.misc.constant(bigNum)
                    #print('nup')
                else:
                    #print('yup')
                    for e in self.edges(n):
                        if self.is_subduction_boundary(e):
                            szloc = self.get_bound_loc(e)

                            xFn  = fn.math.abs(self.xFn - szloc)

                            subZoneAbsDistDict[n] = xFn
        else:
            #firt set all keys to the same value
            subZoneAbsDistDict = dict.fromkeys(self.nodes(), fn.misc.constant(bigNum))
            for e in uG.edges():
                if not self.is_subduction_boundary(e):
                    pass
                #fill in any that have a subducting plate attached
                else:
                    szloc = self.get_bound_loc(e)
                    xFn  = fn.math.abs(self.xFn - szloc)
                    for n in e:
                        subZoneAbsDistDict[n] = xFn







        pIdFn = self.plate_id_fn()
        fnSzDistmap = fn.branching.map(fn_key = pIdFn ,
                          mapping = subZoneAbsDistDict )

        return fnSzDistmap



#model velocity functions

#does evaluate_global send result to all procs or just Root
def mid_plate_point_vel(tectModel, plateId, tmUwMap, surfaceY = 1.0):
        midpoint = np.sort(tectModel.get_boundaries(plateId)).mean()
        vel=tmUwMap.velField.evaluate_global([midpoint, surfaceY])
        return vel


#need to add boolean to float type conversion
def plate_integral_vel(tectModel, tmUwMap, maskFn):

    """returns the average velocity in the region(s) given by a maskFn.
    the maskFn can be a list (vector) in which case a vector output is created.
    """


    tWalls=tectModel.mesh.specialSets["MaxJ_VertexSet"]
    _surfLength  = uw.utils.Integral( maskFn, mesh=tectModel.mesh,
                                     integrationType='Surface', surfaceIndexSet=tWalls)
    surfLength = _surfLength.evaluate()

    _surfVel  = uw.utils.Integral( maskFn*tmUwMap.velField[0], mesh=tectModel.mesh,
                                     integrationType='Surface', surfaceIndexSet=tWalls)
    surfVel = _surfVel.evaluate()

    return np.array(surfVel)/np.array(surfLength)

    pass


#Strain rate query


def strain_rate_min_max(tectModel, tmUwMap, maskFn):

    """
    Note thet the fn.view.min_max() is cumulative, unless reset is called.
    Here we rebuild the min_max() getter, so reset not required.
    """

    sym_strainRate = fn.tensor.symmetric(
                            tmUwMap.velField.fn_gradient )

    vxx = sym_strainRate[0]
    tWalls=tectModel.mesh.specialSets["MaxJ_VertexSet"]

    ##########################
    #currently a min_global_auxiliary() call on a proc with no tWalls is crashing
    ##exploring a workaround, once this bug is fixed, back this out (i.e evaluate on tWalls)
    tWallsSwarm = uw.swarm.Swarm( mesh=tectModel.mesh )
    swarmCoords = tectModel.mesh.data[tWalls.data]
    tWallsSwarm.add_particles_with_coordinates(swarmCoords)
    if not tWalls.data.shape[0]:
        dumCoord = np.column_stack((tectModel.mesh.data[:,0].mean(), tectModel.mesh.data[:,1].mean()))
        tWallsSwarm.add_particles_with_coordinates(np.array(dumCoord))
    ##########################

    srLocMins = []
    srLocMaxs = []

    if not hasattr(maskFn,'__iter__'):

        _szVelGrads = fn.view.min_max(vxx*maskFn, fn_auxiliary=tectModel.xFn)
        dummyFn = _szVelGrads.evaluate(tWallsSwarm )
        minSr = _szVelGrads.min_global()
        maxSr = _szVelGrads.max_global()
        minXVal = _szVelGrads.min_global_auxiliary()[0][0]
        maxXVal = _szVelGrads.max_global_auxiliary()[0][0]

        srLocMins.append((minSr, minXVal))
        srLocMaxs.append((maxSr, maxXVal))
    else:
        # if a list or maskFns
        for mfn in maskFn:
            _szVelGrads = fn.view.min_max(vxx*mfn, fn_auxiliary=tectModel.xFn)
            dummyFn = _szVelGrads.evaluate(tWallsSwarm)
            minSr = _szVelGrads.min_global()
            maxSr = _szVelGrads.max_global()
            minXVal = _szVelGrads.min_global_auxiliary()[0][0]
            maxXVal = _szVelGrads.max_global_auxiliary()[0][0]

            srLocMins.append((minSr, minXVal))
            srLocMaxs.append((maxSr, maxXVal))
    return srLocMins, srLocMaxs


def get_boundary_vel_update(tectModel, platePair, time, dt):
    bv = 0.
    try:
        bv = tectModel.bound_velocity(platePair, time=time)
    except:
        pass

    dx = bv*dt
    newx = (tectModel.get_bound_loc(platePair) + dx)

    return newx


def strain_rate_field_update(tectModel, e, tmUwMap, dist):
    #limit the search radius
    maskFn = tectModel.plate_boundary_mask_fn(dist, out='num',bound=e )
    srLocMins, srLocMaxs = strain_rate_min_max(tectModel, tmUwMap, maskFn)
    if tectModel.is_subduction_boundary(e):
        return srLocMins[0][1]
    else:
        return srLocMaxs[0][1]
