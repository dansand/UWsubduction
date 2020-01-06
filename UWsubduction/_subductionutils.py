import numpy as np
import math
from easydict import EasyDict as edict
from UWsubduction.UWsubduction.interfaces import interface2D , interface_collection
from underworld import function as fn


def cosine_taper(increasingFn, startval, width):
    cosinusoid= 0.5*(1. - fn.math.cos((np.pi*(increasingFn - startval))/(width)) )
    taperFn = fn.branching.conditional( ((increasingFn < startval, 0. ),
                                            (increasingFn > (startval + width), 1. ),
                                           (True,                      cosinusoid)  ))
    return taperFn


def slab_top(trench, normal, gradientFn, ds, maxDepth, mesh):

    """
    Create points representing the top of a slab from trench to maxDepth
    Parameter
    ---------
    trench : list or list like
            Points represnting trench location,
    normal: list or list like
            vector in the horizontal plane normal to trench
    gradientFn: function
             function that returns the dip or the slab dz/ds
             where s is the distance along the normal vector
    ds: float
            distance between points, in model coordinates
    max depth: float, or list or list like
            Maximum depth of slab
    mesh: uw 2 mesh
    """

    #convert everything to numpy arrays
    trench = np.array(trench)
    normal = np.array(normal)/np.linalg.norm(normal)
    maxDepth = np.array(maxDepth)

    #test if gradientFn is a function
    points = []
    points.append(list(trench))

    #set starting values
    #normal/= np.linalg.norm(normal)#unitize
    vertical = np.zeros(mesh.dim)
    vertical[-1] = -1.

    P0 = trench.copy()
    F = gradientFn(0.)
    #print(F)
    H = 0.
    V = 0.
    #print(normal, F)
    S = normal.copy()

    S[-1] = F     #assumes the last component is vertical
    S/= np.linalg.norm(S) #unitize


    while V < maxDepth:

        #Get next set of points
        P1 = P0 + S*ds
        points.append(list(P1))
        P0 = P1

        #update, H, V, F, S
        H +=  np.dot(S*ds, normal)
        V +=  abs(((S*ds)[-1]))
        F = gradientFn(H)
        S = normal.copy()
        S[-1] = F     #assumes the last component is vertical
        S/= np.linalg.norm(S) #unitize




    return(np.array(points))##


#Some common functions for setting up the slab top

def linearGradientFn(S):
    return np.tan(np.deg2rad(-45.))


#def circGradientFn(S):
#    if S == 0.:#
#        return 0.
#    elif S < ndp.radiusOfCurv:
#        return max(-S/np.sqrt((ndp.radiusOfCurv**2 - S**2)), -1e3)
#    else:
#        return -1e5


#def polyGradientFn(S):
#    if S == 0.:
#        return 0.
#    else:
#        return -1*(S/ndp.radiusOfCurv)**2




def tm_uw_map(tempField, velField, swarm, signedDistanceVar, proxyTempVar, proximityVar):

    mapDict = edict()
    mapDict['tempField'] = tempField
    mapDict['velField'] = velField
    mapDict['swarm'] = swarm
    mapDict['signedDistanceVar'] = signedDistanceVar
    mapDict['proxyTempVar'] = proxyTempVar
    mapDict['proximityVar'] = proximityVar
    return mapDict

def build_slab_distance(tectModel, plates, gradFn, maxDepth, tmUwMap):


    """
    We're going to build the slab temperature perturbation by mapping from a interface2D to the swarm
    """
    #tmUwMap


    assert tectModel.is_subduction_boundary((plates[0], plates[1])), 'not a subduction boundary'

    #determine subduction zone location, direction, etc
    szloc = tectModel.undirected[plates[0]][plates[1]]['loc']             #Location
    dir_ = tectModel.subduction_direction((plates[0], plates[1]))           #Direction facing
    normal = [dir_,0.]                                       #Direction facing (as a vector)

    spId = tectModel.subduction_edge_order((plates[0], plates[1]))[0]         #subduction plate Id
    slabage = tectModel.undirected[plates[0]][plates[1]]['ages'][spId]      #subduction plate Age at the trench
    slabthickness = 2.32*math.sqrt(1.*slabage )              #Diffusive thickness, dimensionless
    plateBounds = np.sort(tectModel.get_boundaries(spId))
    insidePt = (np.array(plateBounds).mean(), 1 - maxDepth*5)




    ds = (tectModel.maxX - tectModel.minX)/(5.*tectModel.mesh.elementRes[0])     #distance between points
    #make the slab data
    slabdata = slab_top([szloc, 1.0], normal, gradFn, ds, maxDepth, tectModel.mesh) #Build the top of slab line
    #return slabdata



    #build the interface2D
    id_ = -99 #in this case, just a dummy value
    slabLine = interface2D(tectModel.mesh, tmUwMap.velField, slabdata[:,0], slabdata[:,1],
                            slabthickness, id_, insidePt=insidePt)

    #print (1. - slabLine.swarm.particleCoordinates.data[:,1].min())*2900

    #map to signed distance

    sd, pts = slabLine.compute_signed_distance(tmUwMap.swarm.particleCoordinates.data, distance=2.*slabthickness)
    tmUwMap.signedDistanceVar.data[np.logical_and(sd>0, sd<=slabLine.thickness)] = \
    sd[np.logical_and(sd>0, sd<=slabLine.thickness)]

    #Now apply the depth cutoff
    tmUwMap.signedDistanceVar.data[tmUwMap.swarm.particleCoordinates.data[:,1] < (1. - maxDepth)] = 0.0




def build_slab_temp(tmUwMap, potentialTemp, slabAge):

    #map to proxyTempVariable

    #a numpy array
    slabTempProx  = potentialTemp*fn.math.erf((tmUwMap.signedDistanceVar)/(2.*np.sqrt(1.*slabAge)))


    conditions = [ ( slabTempProx > 0.1, slabTempProx ),
                   ( True, tmUwMap.proxyTempVar)]

    tmUwMap.proxyTempVar.data[:] = fn.branching.conditional( conditions ).evaluate(tmUwMap.swarm)


def build_fault(tectModel, plates, gradFn, thickness, maxDepth, ds, vertoffset, tmUwMap):


    """
    We're going to build the slab temperature perturbation by mapping from a interface2D to the swarm
    """

    assert tectModel.is_subduction_boundary((plates[0], plates[1])), 'not a subduction boundary'

    #determine subduction zone location, direction, etc
    szloc = tectModel.undirected[plates[0]][plates[1]]['loc']             #Location
    dir_ = tectModel.subduction_direction((plates[0], plates[1]))           #Direction facing
    normal = [dir_,0.]                                       #Direction facing (as a vector)

    spId = tectModel.subduction_edge_order((plates[0], plates[1]))[0]         #subduction plate Id
    slabage = tectModel.undirected[plates[0]][plates[1]]['ages'][spId]      #subduction plate Age at the trench
    slabthickness = 2.32*math.sqrt(1.*slabage )              #Diffusive thickness, dimensionless
    plateBounds = np.sort(tectModel.get_boundaries(spId))
    insidePt = (np.array(plateBounds).mean(), 1 - maxDepth*5)



    #make the slab data,
    #use maxDepth - vertoffset - ds, as the vertical distance is relative to the starting depth
    slabdata = slab_top([szloc, 1.0 - vertoffset], normal, gradFn, ds, maxDepth - vertoffset - ds, tectModel.mesh)

    #plate bounds are sorted l-r so we include ds to get the end point
    plateDataXs = np.arange(plateBounds[0] + ds, plateBounds[1] + ds, ds)

    plateDataYs = (1.0 - vertoffset)*np.ones(len(plateDataXs))
    plateData = np.column_stack((plateDataXs, plateDataYs))

    #exclude the first point in the slab top, as it appears in plate data
    faultData = np.row_stack((plateData, slabdata[1:,:]))

    #build the interface2D
    fault = interface2D(tectModel.mesh, tmUwMap.velField, faultData[:,0], faultData[:,1],
                            thickness, spId, insidePt=insidePt)

   # with fault.swarm.deform_swarm():
   #     fault.swarm.particleCoordinates.data[:] += fault.director.data*offset

    return fault


def remove_faults_from_boundaries(tectModel, fCollect,  maskFn, out=(999999.,999999.)):

    """
    Remove Fauls from Ridges
    """
    #build the main plate ID function, including the plate boundary mask.
    #we need to flip the mask function before handing to plate_id_fn

    maskFn_ = tectModel.t2f(maskFn)
    pIdFn = tectModel.plate_id_fn(maskFn=maskFn_)
    #evaluate on the master swarm (local proc)
    #iDs = pIdFn.evaluate(masterSwarm)
    #loop through faults
    for f in fCollect:
        #mask for the plate, inclusing the specified plate boundary mask
        #mask1 = np.where(iDs == f.ID)[0]
        mask1 = np.where(pIdFn.evaluate(f.swarm) == f.ID)[0]  #can use == 0

        with f.swarm.deform_swarm():
            f.swarm.particleCoordinates.data[mask1] = out
            #f.rebuild() #10/02/18

#    for f in fCollect:
#        mask = np.where(maskFn.evaluate(f.swarm) == True)[0]  #can use == 0
#        with f.swarm.deform_swarm():
#            f.swarm.particleCoordinates.data[mask] = out

#        f.rebuild() #10/02/18


def remove_fault_drift(fCollect, verticalLevel, tolFac =1e-2 ):

    """
    Stop fault particles from drifting aboove their starting position
    """

    for f in fCollect:
        yPosArray = np.ones(len(f.swarm.particleCoordinates.data[:,1]))*verticalLevel
        mask = np.where(f.swarm.particleCoordinates.data[:,1] > verticalLevel + (1. - verticalLevel)*tolFac)[0]
        with f.swarm.deform_swarm():
            f.swarm.particleCoordinates.data[mask,1] = yPosArray[mask]
        f.rebuild() #10/02/18


def pop_or_perish(tectModel, fCollect, masterSwarm, maskFn, ds):

    """
    Adds particles from a 'template' array; assumed to be a uniform layout near the surface
    Akin to adding more `weak crust` on the subducting plate
    Need to be careful to properly catch empty arrays,
    Otherwise parallel becomes a problem
    masterSwarm is the surface layout of the faults.
    """

    #build the main plate ID function, including the plate boundary mask.
    #we need to flip the mask function before handing to plate_id_fn

    maskFn_ = tectModel.t2f(maskFn)
    pIdFn = tectModel.plate_id_fn(maskFn=maskFn_)
    #evaluate on the master swarm (local proc)
    iDs = pIdFn.evaluate(masterSwarm)
    #loop through faults
    for f in fCollect:
        #mask for the plate, inclusing the specified plate boundary mask
        mask1 = np.where(iDs == f.ID)[0]


        plateParticles = masterSwarm.particleCoordinates.data[mask1,:]
        #this is hopefully the only parallel safeguard we need. But be afaid.

        #initially I was using a KDTree query to apply the new fault particles.
        #But this is really very unstable
        #if not f.empty:
        #    mask3 = (f.kdtree.query(plateParticles)[0] > ds)
        #    dataToAdd = masterSwarm.particleCoordinates.data[mask1,:][mask3]
            #now we can add these in
        #    f.swarm.add_particles_with_coordinates(dataToAdd)

        #Now I'm using the global extent of the interface2D swarm,
        #To define where NOT to add new particles
        minmax_coordx = fn.view.min_max(fn.coord()[0])
        ignore = minmax_coordx.evaluate(f.swarm)
        leftExt = minmax_coordx.min_global()
        rightExt = minmax_coordx.max_global()
        #print("fault extent", leftExt , rightExt)

        if not f.empty:
            mask3 = np.logical_or(plateParticles[:,0] < (leftExt - ds),  plateParticles[:,0] > (rightExt + ds))
            dataToAdd = masterSwarm.particleCoordinates.data[mask1,:][mask3]
            f.swarm.add_particles_with_coordinates(dataToAdd)
            #print("adding data", dataToAdd.shape)

        f.rebuild() #10/02/18
