
# coding: utf-8

# # A fully dynamic model (no temperature diffusion)
# 
# 
# In this notebook, we set up an arbitrary scenario, involving two subduction zones. This notebook is intended to demonstate:
# 
# * how the TectModel object faciliates building initial conditions
# * how to update the TectModel using kinematic information from the evolving Underworld2 model
# * How to 'manage' faults using the TectModel
# 
# 

# In[1]:


#load in parent stuff

import nb_load_stuff
from tectModelClass import *


# In[2]:


#If run through Docker we'll point at the local 'unsupported dir.'
#On hpc, the path should also include a directory holding the unsupported_dan.
import os
import sys

#this does't actually need to be protected. More a reminder it's an interim measure
try:
    sys.path.append('../../unsupported')
except:
    pass


# In[3]:


#%load_ext autoreload
from unsupported_dan.UWsubduction.base_params import *
from unsupported_dan.UWsubduction.subduction_utils import *
from unsupported_dan.interfaces.marker2D import markerLine2D, line_collection
from unsupported_dan.interfaces.smoothing2D import *


#reload(base_params)


# In[4]:


import numpy as np
import underworld as uw
from underworld import function as fn
import glucifer
from easydict import EasyDict as edict
import networkx as nx
import operator




# In[5]:


ndp.maxDepth


# ## Changes to base params

# In[72]:


#These will keep changing if the notebook is run again without restarting!

ndp.depth *= 0.8 #800 km
ndp.faultThickness *= 1.5 #15 km
ndp.interfaceViscCutoffDepth *= 1.5 #150 km
ndp.maxDepth *= 1.5
md.res = 64
ndp.radiusOfCurv*=0.72  #~250 km
md.nltol = 0.025
md.ppc = 20
#print(ndp.faultThickness*2900)
ndp.yieldStressMax *=0.5  #150 Mpa

#this flag currently doesn't appear in the base_params
md.periodic = False


# ## Build mesh, Stokes Variables

# In[73]:


#(ndp.rightLim - ndp.leftLim)/ndp.depth
#md.res = 64


# In[74]:


yres = int(md.res)
xres = int(md.res*12) 

mesh_periodic    = [False, False]
if md.periodic:
    mesh_periodic    = [True, False]
    

mesh = uw.mesh.FeMesh_Cartesian( elementType = (md.elementType),
                                 elementRes  = (xres, yres), 
                                 minCoord    = (ndp.leftLim, 1. - ndp.depth), 
                                 maxCoord    = (ndp.rightLim, 1.),
                                  periodic    = mesh_periodic ) 

velocityField = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=2)
pressureField   = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )


velocityField.data[:] = 0.
pressureField.data[:] = 0.


# ## Build plate model

# In[9]:


#Set up some velocityies
cm2ms = (1/100.)*(1./(3600*24*365)) 

v1= 2.*cm2ms #m/s
v1 /= sf.velocity

v2= -2.*cm2ms #
v2 /= sf.velocity



ma2s = 1e6*(3600*24*365)
endTime = 20*ma2s/sf.time
dt = 0.1*ma2s/sf.time
testTime = 5*ma2s/sf.time


# In[10]:


#20 Ma moddel, timestep of 200 Ka 
tg = TectModel(mesh, 0, endTime, dt)

tg.add_plate(1)
tg.add_plate(2)
tg.add_plate(3)


# In[11]:


tg.add_left_boundary(1)
tg.add_subzone(1, 2, ndp.subZoneLoc, subInitAge=ndp.slabMaxAge, upperInitAge=ndp.opMaxAge)
tg.add_subzone(3, 2., 0.4, subInitAge=ndp.slabMaxAge, upperInitAge=ndp.opMaxAge)

tg.add_right_boundary(3, 0.)


# ## Build plate age

# In[12]:


pIdFn = tg.plate_id_fn()
pAgeDict = tg.plate_age_fn() 

fnAge_map = fn.branching.map(fn_key = pIdFn , 
                          mapping = pAgeDict )

#fig = glucifer.Figure(figsize=(600, 300))
#fig.append( glucifer.objects.Surface(tg.mesh, fnAge_map ))
#fig.show()


# In[13]:


coordinate = fn.input()
depthFn = mesh.maxCoord[1] - coordinate[1]

platethickness = 2.32*fn.math.sqrt(1.*fnAge_map )  

halfSpaceTemp = ndp.potentialTemp*fn.math.erf((depthFn)/(2.*fn.math.sqrt(1.*fnAge_map)))

plateTempProxFn = fn.branching.conditional( ((depthFn > platethickness, ndp.potentialTemp ), 
                                           (True,                      halfSpaceTemp)  ))



# In[16]:


#fig = glucifer.Figure(figsize=(600, 300))
#fig.append( glucifer.objects.Surface(tg.mesh, plateTempProxFn))
#fig.show()


# ## Make swarm and Slabs

# In[17]:


def circGradientFn(S):
    if S == 0.:
        return 0.
    elif S < ndp.radiusOfCurv:
        return max(-S/np.sqrt((ndp.radiusOfCurv**2 - S**2)), -1e3)
    else:
        return -1e5


# In[18]:


swarm = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
layout = uw.swarm.layouts.PerCellRandomLayout(swarm=swarm, particlesPerCell=int(md.ppc))
swarm.populate_using_layout( layout=layout ) # Now use it to populate.
proxyTempVariable = swarm.add_variable( dataType="double", count=1 )
proximityVariable      = swarm.add_variable( dataType="int", count=1 )
signedDistanceVariable = swarm.add_variable( dataType="double", count=1 )

#
proxyTempVariable.data[:] = 1.0
proximityVariable.data[:] = 0.0
signedDistanceVariable.data[:] = 0.0


# In[19]:


#All of these wil be needed by the slab / fault setup functions
#We have two main options, bind them to the TectModel class. 
#or provide them to the functions
#collection them in a dictionary may be a useful way too proviede them to the function 
#wthout blowing out the function arguments

tmUwMap = tm_uw_map([], velocityField, swarm, 
                    signedDistanceVariable, proxyTempVariable, proximityVariable)




# In[20]:


#define fault particle spacing, here ~5 paricles per element
ds = (tg.maxX - tg.minX)/(2.*tg.mesh.elementRes[0])

fCollection = line_collection([])




for e in tg.undirected.edges():
    if tg.is_subduction_boundary(e):
        build_slab_distance(tg, e, circGradientFn, ndp.maxDepth, tmUwMap)        
        fb = build_fault(tg, e, circGradientFn, ndp.faultThickness , ndp.maxDepth, ds, ndp.faultThickness, tmUwMap)
        fCollection.append(fb)

#
build_slab_temp(tmUwMap, ndp.potentialTemp, ndp.slabMaxAge)
fnJointTemp = fn.misc.min(proxyTempVariable,plateTempProxFn)

#And now reevaluate this guy on the swarm
proxyTempVariable.data[:] = fnJointTemp.evaluate(swarm)


# In[21]:


#ndp.maxDepth


# In[23]:


#fig = glucifer.Figure(figsize=(600, 300))
#fig.append( glucifer.objects.Points(swarm, proxyTempVariable))
    

#fig.show()
#fig.save_database('test.gldb')


# ##  Fault rebuild

# In[24]:


# Setup a swarm to define the replacment positions

fThick= fCollection[0].thickness

faultloc = 1. - ndp.faultThickness*md.faultLocFac

allxs = np.arange(mesh.minCoord[0], mesh.maxCoord[0], ds )[:-1]
allys = (mesh.maxCoord[1] - fThick)*np.ones(allxs.shape)

faultMasterSwarm = uw.swarm.Swarm( mesh=mesh )
dummy =  faultMasterSwarm.add_particles_with_coordinates(np.column_stack((allxs, allys)))
del allxs
del allys


# In[26]:


#ridgedist = 400e3/sf.lengthScale
#subdist = 150e3/sf.lengthScale


#constant width Mask Fns. Not very useful
#ridgeMaskFn = tg.ridge_mask_fn(ridgedist)
#subMaskFn = tg.subduction_mask_fn(subdist)
#boundMaskFn = tg.combine_mask_fn(ridgeMaskFn , subMaskFn )


#dummy = remove_faults_from_boundaries(fCollection, ridgeMaskFn)
#dummy = remove_fault_drift(fCollection, faultloc)
#dummy = pop_or_perish(tg, fCollection, faultMasterSwarm, boundMaskFn, ds)



ridgeMaskFn = tg.variable_boundary_mask_fn(distMax=100., distMin=0.0, relativeWidth = 0.1, 
                                  minPlateLength =50e3/sf.lengthScale,  
                                           out = 'bool', boundtypes='ridge')

boundMaskFn = tg.plate_interior_mask_fn(relativeWidth=0.8, 
                                        minPlateLength=10e3/sf.lengthScale, invert=False)


dummy = remove_faults_from_boundaries(tg, fCollection, ridgeMaskFn)
dummy = remove_fault_drift(fCollection, faultloc)
dummy = pop_or_perish(tg, fCollection, faultMasterSwarm, boundMaskFn, ds)


# In[28]:


#fig = glucifer.Figure(figsize=(400, 200))
#fig.append( glucifer.objects.Surface(tg.mesh, boundMaskFn))
#for f in fCollection:
#    fig.append( glucifer.objects.Points(f.swarm, pointSize=5))

#fig.show()


# ## Proximity
# 
# 

# In[29]:


proximityVariable.data[:] = 0


# In[30]:


for f in fCollection:
    f.rebuild()
    f.set_proximity_director(swarm, proximityVariable, searchFac = 2., locFac=1.0)


# In[32]:


#figProx = glucifer.Figure(figsize=(960,300) )
#figProx.append( glucifer.objects.Points(swarm , proximityVariable))
#for f in fCollection:
#    figProx.append( glucifer.objects.Points(f.swarm, pointSize=5))
#figProx.show()


# In[71]:





# ## Boundary conditions

# In[33]:


iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]

tWalls = mesh.specialSets["MaxJ_VertexSet"]
bWalls = mesh.specialSets["MinJ_VertexSet"]

#if periodicc, we're goinf to fix a node at the base. 
if md.periodic:
    lWalls = mesh.specialSets["MinI_VertexSet"]
    fixNode = tWalls & lWalls 
    velnodeset = mesh.specialSets["Empty"]
    velnodeset += fixNode


velBC  = uw.conditions.DirichletCondition( variable        = velocityField, 
                                           indexSetsPerDof = (iWalls + velnodeset, jWalls) )


# ## Bouyancy

# In[34]:


# Now create a buoyancy force vector using the density and the vertical unit vector. 
thermalDensityFn = ndp.rayleigh*(1. - proxyTempVariable)

gravity = ( 0.0, -1.0 )

buoyancyMapFn = thermalDensityFn*gravity


# ## Rheology

# In[35]:


symStrainrate = fn.tensor.symmetric( 
                            velocityField.fn_gradient )

#Set up any functions required by the rheology
strainRate_2ndInvariant = fn.tensor.second_invariant( 
                            fn.tensor.symmetric( 
                            velocityField.fn_gradient ))



def safe_visc(func, viscmin=ndp.viscosityMin, viscmax=ndp.viscosityMax):
    return fn.misc.max(viscmin, fn.misc.min(viscmax, func))


# In[36]:


temperatureFn = proxyTempVariable
adiabaticCorrectFn = depthFn*ndp.tempGradMantle
dynamicPressureProxyDepthFn = pressureField/sf.pressureDepthGrad
druckerDepthFn = fn.misc.max(0.0, depthFn + md.druckerAlpha*(dynamicPressureProxyDepthFn))

#Diffusion Creep
diffusionUM = (1./ndp.diffusionPreExp)*            fn.math.exp( ((ndp.diffusionEnergy + (depthFn*ndp.diffusionVolume))/((temperatureFn+ adiabaticCorrectFn + ndp.surfaceTemp))))

diffusionUM =     safe_visc(diffusionUM)
    
diffusionLM = ndp.lowerMantleViscFac*(1./ndp.lowerMantlePreExp)*            fn.math.exp( ((ndp.lowerMantleEnergy + (depthFn*ndp.lowerMantleVolume))/((temperatureFn+ adiabaticCorrectFn + ndp.surfaceTemp))))

diffusionLM =     safe_visc(diffusionLM)

    
#combine upper and lower mantle   
mantleCreep = fn.branching.conditional( ((depthFn < ndp.lowerMantleDepth, diffusionUM ), 
                                           (True,                      diffusionLM )  ))

#Define the mantle Plasticity
ys =  ndp.cohesionMantle + (druckerDepthFn*ndp.frictionMantle)
ysf = fn.misc.min(ys, ndp.yieldStressMax)
yielding = ysf/(2.*(strainRate_2ndInvariant) + 1e-15) 


mantleRheologyFn = safe_visc(fn.misc.min(mantleCreep, yielding), viscmin=ndp.viscosityMin, viscmax=ndp.viscosityMax)

#Subduction interface viscosity
interfaceViscosityFn = fn.misc.constant(0.5)


# In[37]:


#viscconds = ((proximityVariable == 0, mantleRheologyFn),
#             (True, interfaceViscosityFn ))

#viscosityMapFn = fn.branching.conditional(viscconds)
#viscosityMapFn = mantleRheologyFn


viscosityMapFn = fn.branching.map( fn_key = proximityVariable,
                             mapping = {0:mantleRheologyFn,
                                        1:interfaceViscosityFn,
                                       3:interfaceViscosityFn} )


# ## Stokes

# In[38]:


surfaceArea = uw.utils.Integral(fn=1.0,mesh=mesh, integrationType='surface', surfaceIndexSet=tWalls)
surfacePressureIntegral = uw.utils.Integral(fn=pressureField, mesh=mesh, integrationType='surface', surfaceIndexSet=tWalls)

NodePressure = uw.mesh.MeshVariable(mesh, nodeDofCount=1)
Cell2Nodes = uw.utils.MeshVariable_Projection(NodePressure, pressureField, type=0)
Nodes2Cell = uw.utils.MeshVariable_Projection(pressureField, NodePressure, type=0)

def smooth_pressure(mesh):
    # Smooths the pressure field.
    # Assuming that pressure lies on the submesh, do a cell -> nodes -> cell
    # projection.

    Cell2Nodes.solve()
    Nodes2Cell.solve()

# a callback function to calibrate the pressure - will pass to solver later
def pressure_calibrate():
    (area,) = surfaceArea.evaluate()
    (p0,) = surfacePressureIntegral.evaluate()
    offset = p0/area
    pressureField.data[:] -= offset
    smooth_pressure(mesh)


# In[39]:


stokes = uw.systems.Stokes( velocityField  = velocityField, 
                                   pressureField  = pressureField,
                                   conditions     = [velBC,],
                                   fn_viscosity   = viscosityMapFn, 
                                   fn_bodyforce   = buoyancyMapFn )


# In[40]:


solver = uw.systems.Solver(stokes)

solver.set_inner_method("mumps")
solver.options.scr.ksp_type="cg"
solver.set_penalty(1.0e7)
solver.options.scr.ksp_rtol = 1.0e-4


# In[41]:


solver.solve(nonLinearIterate=True, nonLinearTolerance=md.nltol, callback_post_solve = pressure_calibrate)
solver.print_stats()


# ## Test

# In[42]:


#fault = markerLine2D(tg.mesh, tmUwMap.velField, [], [],
#                           0.01, 7, insidePt=(0.0, 0.8))

#f = fCollection[0]
#iDs = pIdFn.evaluate(faultMasterSwarm)
#mask1 = np.where(iDs == 7)[0]   #not a problem with empty array returned by 
#mask1 = [1,2]

#plateParticles = faultMasterSwarm.particleCoordinates.data[mask1,:] #not a problem with empty array returned by 
#mask3 = (f.kdtree.query(plateParticles)[0] > ds)


# In[43]:


#plateParticles.shape[0] 


# In[44]:


#if plateParticles.shape[0] > 0:
#    mask3 = (f.kdtree.query(plateParticles)[0] > ds)


# In[45]:


#faultMasterSwarm[1,2][maks3]


# ## Swarm Advector

# In[46]:


advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=velocityField, order=2 )


# In[47]:


population_control = uw.swarm.PopulationControl(swarm, deleteThreshold=0.006, 
                                                splitThreshold=0.25,maxDeletions=1, maxSplits=3, aggressive=True,
                                                aggressiveThreshold=0.9, particlesPerCell=int(md.ppc))


# ## Update functions

# In[48]:


# Here we'll handle everything that should be advected - i.e. every timestep
def advect_update():
    # Retrieve the maximum possible timestep for the advection system.
    dt = advector.get_max_dt()
    # Advect swarm
    advector.integrate(dt)
    
    #Advect faults
    for f in fCollection:
        f.advection(dt)
    
    
    return dt, time+dt, step+1


# In[51]:


def update_faults():
    
    ##the mask fns are static at this stage
    #ridgeMaskFn = tg.ridge_mask_fn(ridgedist)
    #subMaskFn = tg.subduction_mask_fn(subdist)
    #boundMaskFn = tg.combine_mask_fn(ridgeMaskFn , subMaskFn )
    
    
    dummy = remove_faults_from_boundaries(tg, fCollection, ridgeMaskFn)
    dummy = remove_fault_drift(tg, fCollection, faultloc, tolFac =ds*2)
    dummy = pop_or_perish(tg, fCollection, faultMasterSwarm, boundMaskFn, ds*3)
    
    for f in fCollection:
        
        #Remove particles below a specified depth
        depthMask = f.swarm.particleCoordinates.data[:,1] <         (1. - (ndp.interfaceViscCutoffDepth - ndp.faultThickness))
        with f.swarm.deform_swarm():
            f.swarm.particleCoordinates.data[depthMask] = (9999999., 9999999.)
        
        #Here we're grabbing a 'black box' routine , 
        #which is supposed to maintain particle density and smooth
        #quite experimental!!!
        repair_markerLines(f, ds, k=8)
    


# In[52]:


#update_faults()


# In[53]:


def update_swarm():
    
    population_control.repopulate()
    
    for f in fCollection:
        f.rebuild()
        f.set_proximity_director(swarm, proximityVariable, searchFac = 2., locFac=1.0,
                                maxDistanceFn=fn.misc.constant(2.))
        
    #A simple depth cutoff for proximity
    depthMask = swarm.particleCoordinates.data[:,1] < (1. - ndp.interfaceViscCutoffDepth)
    proximityVariable.data[depthMask] = 0
    
    
    


# In[54]:


def boundary_ridge_update(tectModel, tmUwMap, e, dt):    
    
    #get the avergae velocity of the plate
    maskFn = tectModel.plate_interior_mask_fn(relativeWidth=0.8, plate = e[0],  out='num')
    velx = plate_integral_vel(tectModel, tmUwMap, maskFn)
    dx = velx*dt
    return dx


def strain_rate_field_update(tectModel, e, tmUwMap):
    dist = 100e3/sf.lengthScale
    maskFn = tectModel.plate_boundary_mask_fn(dist, out='num',bound=e )
    srLocMins, srLocMaxs = strain_rate_min_max(tectModel, tmUwMap, maskFn)
    if tg.is_subduction_boundary(e):
        return srLocMins[0][1]
    else:
        return srLocMaxs[0][1]
    

def update_tect_model(tectModel, tmUwMap, dt = 0.0 ):
    
    """
    An example of how we can update the tect_model
    """
    for e in tectModel.undirected.edges():
        if e[0] == e[1]:
            dx = boundary_ridge_update(tectModel,tmUwMap, e, dt)
            newx = (tectModel.get_bound_loc(e) + dx)[0]
            tectModel.set_bound_loc(e, newx)
        elif tectModel.is_subduction_boundary(e):
            e = tectModel.subduction_edge_order(e)
            newx = strain_rate_field_update(tectModel, e, tmUwMap)
            tectModel.set_bound_loc(e, newx)
        else:
            pass
        
        
def rebuild_mask_fns():
    
    #ridgeMaskFn = tg.ridge_mask_fn(ridgedist)
    #subMaskFn = tg.subduction_mask_fn(subdist)
    #boundMaskFn = tg.combine_mask_fn(ridgeMaskFn , subMaskFn )
    #pIdFn = tg.plate_id_fn() #just here for Viz
    #return ridgeMaskFn, subMaskFn, boundMaskFn, pIdFn
    
    ridgeMaskFn = tg.variable_boundary_mask_fn(distMax=100., distMin=0.0, relativeWidth = 0.1, 
                                      minPlateLength =50e3/sf.lengthScale,  
                                               out = 'bool', boundtypes='ridge')

    boundMaskFn = tg.plate_interior_mask_fn(relativeWidth=0.8, 
                                            minPlateLength=10e3/sf.lengthScale, invert=False)
                
       
    return ridgeMaskFn, boundMaskFn


# In[55]:


#ridgeMaskFn, subMaskFn, boundMaskFn = rebuild_mask_fns()


# In[56]:


#update_tect_model(tg, tmUwMap,dttest)


# In[57]:


#f.data.shape


# In[58]:


#update_faults()
#update_swarm()


# ## Track the values of the plate bounaries

# In[61]:


valuesDict = edict({})
valuesDict.timeAtSave = []
valuesDict.stepAtSave = []
for e in tg.undirected.edges():
    valuesDict[str(e)] = []
valuesDict    


# In[62]:


def valuesUpdateFn():
    
    """ 
    Assumes global variables:
    * time
    * step 
    ...
    + many functions
    """
    
    
    #save the time and step
    valuesDict.timeAtSave.append(time) 
    valuesDict.stepAtSave.append(step)
    
    for e in tg.undirected.edges():
        if tg.is_subduction_boundary(e):
            ee = tg.subduction_edge_order(e) #hacky workaround for the directed/ undireted. need get_bound_loc
        else:
            ee = e

        valuesDict[str(e)].append(tg.get_bound_loc(ee))
        
        
    #save
    if uw.rank()==0:
        fullpath = os.path.join(outputPath + "tect_model_data")
        #the '**' is important
        np.savez(fullpath, **valuesDict)
    


# In[63]:


#valuesUpdateFn()
#valuesDict  
#!ls output


# ## Viz

# In[64]:


outputPath = os.path.join(os.path.abspath("."),"output/")

if uw.rank()==0:
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
uw.barrier()


# In[65]:


store1 = glucifer.Store('output/subduction1')
store2 = glucifer.Store('output/subduction2')
#store3 = glucifer.Store('output/subduction3')


figProx = glucifer.Figure(store1, figsize=(960,300) )
figProx.append( glucifer.objects.Points(swarm , proximityVariable))
for f in fCollection:
    figProx.append( glucifer.objects.Points(f.swarm, pointSize=5))
#figProx.show()

figVisc = glucifer.Figure( store2, figsize=(960,300) )
figVisc.append( glucifer.objects.Points(swarm, viscosityMapFn, pointSize=2, logScale=True) )



#figMask = glucifer.Figure( store3, figsize=(960,300) )
#figMask.append( glucifer.objects.Surface(mesh, pIdFn , valueRange=[0,3]) )
#figMask.append( glucifer.objects.Surface(mesh,  boundMaskFn) )



# In[66]:


#figProx.show()
#figProx.save_database('test.gldb')


# In[67]:


#1e-2*2900.


# ## Main Loop

# In[132]:


time = 0.  # Initial time
step = 0 
maxSteps = 1000      # Maximum timesteps (201 is recommended)
steps_output = 10   # output every 10 timesteps
faults_update = 10
dt_model = 0.
steps_update_model = 10

valuesUpdateFn()


# In[47]:


#while time < tg.times[-1]:
while step < maxSteps:
    # Solve non linear Stokes system
    solver.solve(nonLinearIterate=True)
    
    #advect swarm and faults
    dt, time, step =  advect_update()
    dt_model += dt
    
    
    #running fault healing/addition, map back to swarm
    if step % faults_update == 0:
        update_faults()      
        update_swarm()
        
    #update tectonic model
    if step % steps_update_model == 0:
        update_tect_model(tg, tmUwMap, dt = dt_model)
        dt_model = 0.
        #ridgeMaskFn, subMaskFn, boundMaskFn, pIdFn= rebuild_mask_fns()
        
        ridgeMaskFn, boundMaskFn = rebuild_mask_fns()
        valuesUpdateFn()
        
    
    # output figure to file at intervals = steps_output
    if step % steps_output == 0 or step == maxSteps-1:
        #Important to set the timestep for the store object here or will overwrite previous step
        store1.step = step
        store2.step = step
        #store3.step = step
        figProx.save(    outputPath + "proximity"    + str(step).zfill(4))
        figVisc.save(    outputPath + "visc"    + str(step).zfill(4))
        #figMask.save(    outputPath + "mask"    + str(step).zfill(4))
    
    if uw.rank()==0:
        print 'step = {0:6d}; time = {1:.3e}'.format(step,time)


# In[48]:


np.arange(1, 4)

