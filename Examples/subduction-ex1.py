
# coding: utf-8

# # A fully dynamic model (no temperature diffusion)
# 
# 
# In this notebook, we set up an arbitrary scenario, involving two subduction zones. This notebook is intended to demonstate:
# 
# * how the TectModel object faciliates building initial conditions
# * how to set up a weak crust (weak layer) approach to the subduction interface
# * to keep this  example simple, we do not evolve thet TectModel, we simple let the Underworld simulation run from the starting conditions...until it breaks
# 
# 

# In[66]:


#load in parent stuff

import nb_load_stuff
from tectModelClass import *


# In[67]:


#If run through Docker we'll point at the local 'unsupported dir.'
#On hpc, the path should also include a directory holding the unsupported_dan.
import os
import sys

#this does't actually need to be protected. More a reminder it's an interim measure
try:
    sys.path.append('../../unsupported')
except:
    pass


# In[68]:


from unsupported_dan.UWsubduction.subduction_utils import *
from unsupported_dan.interfaces.smoothing2D import *
from unsupported_dan.utilities.misc import cosine_taper
from unsupported_dan.interfaces.interface2D import interface2D , interface_collection


#reload(base_params)


# In[69]:


import numpy as np
import underworld as uw
from underworld import function as fn
import glucifer
from easydict import EasyDict as edict
import networkx as nx
import operator




# ## Create Directory Output Structure

# In[70]:


#outputPath = os.path.join(os.path.abspath("."),"output/")
outputPath = os.path.join(os.path.abspath("."),"output/files")

if uw.rank()==0:
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
uw.barrier()


# ## Parameters / Scaling
# 
# * For more information see, `UWsubduction/Background/scaling`
# 

# In[107]:


#import parameters, model settings, unit registry, scaling system, etc

from unsupported_dan.UWsubduction.minimal_example import paramDict_dim, modelDict_dim, UnitRegistry
from unsupported_dan.UWsubduction.default_scaling import sub_scaling, build_nondim_dict
from unsupported_dan.UWsubduction.minimal_example import rayleighNumber, stressScale, pressureDepthGrad


#define some more concise names
ur = UnitRegistry
sca = sub_scaling
ndimlz = sca.nonDimensionalize
#build the dimensionless paramter / model dictionaries
ndp = build_nondim_dict(paramDict_dim  , sca)   
md = build_nondim_dict(modelDict_dim  , sca)

assert ndimlz(paramDict_dim.refLength) == 1.0

# changes to base params (for testing)
md.faultThickness *= 1.5 #15 km
md.res = 48
md.periodic = True #add this item to the dictionary


# ## Build mesh, Stokes Variables

# In[72]:


#(ndp.rightLim - ndp.leftLim)/ndp.depth
#md.res = 64


# In[73]:


yres = int(md.res)
xres = int(md.res*6) 

halfWidth = 0.5*md.depth*md.aspectRatio 

minCoord_    = (-1.*halfWidth, 1. - md.depth) 
maxCoord_    = (halfWidth, 1.)

mesh = uw.mesh.FeMesh_Cartesian( elementType = (md.elementType),
                                 elementRes  = (xres, yres), 
                                 minCoord    = minCoord_, 
                                 maxCoord    = maxCoord_) 

velocityField = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=2)
pressureField   = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )
    

velocityField.data[:] = 0.
pressureField.data[:] = 0.


# ## Build Tectonic Model

# In[74]:


endTime = ndimlz(20.*ur.megayear)
dt = endTime                           #dummy value as we're onlu using the tectModel to set up the initial conds.
tm = TectModel(mesh, 0, endTime, dt)

tm.add_plate(1)
tm.add_plate(2)
tm.add_plate(3)


# In[75]:


tm.add_left_boundary(1)
tm.add_subzone(1, 2, md.subZoneLoc, subInitAge=md.slabAge, upperInitAge=md.opAgeAtTrench)
tm.add_subzone(3, 2., 0.4, subInitAge=md.slabAge, upperInitAge=md.opAgeAtTrench)
tm.add_right_boundary(3, 0.)


# ## Build plate age
# 
# See UWsubduction/Background/scaling for discussion on 
# 
# `potentialTemp_` vs `potentialTemp`

# In[76]:


pIdFn = tm.plate_id_fn()
pAgeDict = tm.plate_age_fn() 

fnAge_map = fn.branching.map(fn_key = pIdFn , 
                          mapping = pAgeDict )

#fig = glucifer.Figure(figsize=(600, 300))
#fig.append( glucifer.objects.Surface(tg.mesh, fnAge_map, onMesh=True ))
#fig.show()


# In[77]:


coordinate = fn.input()
depthFn = mesh.maxCoord[1] - coordinate[1]

platethickness = 2.32*fn.math.sqrt(1.*fnAge_map )  

halfSpaceTemp = ndp.potentialTemp_*fn.math.erf((depthFn)/(2.*fn.math.sqrt(1.*fnAge_map)))

plateTempProxFn = fn.branching.conditional( ((depthFn > platethickness, ndp.potentialTemp_), 
                                           (True,                      halfSpaceTemp)  ))



# In[78]:


#fig = glucifer.Figure(figsize=(600, 300))
#fig.append( glucifer.objects.Surface(tg.mesh, plateTempProxFn, onMesh=True, colourBar=False))
#fig.show()


# ## Make swarm and Slabs

# In[79]:


def circGradientFn(S):
    if S == 0.:
        return 0.
    elif S < md.radiusOfCurv:
        return max(-S/np.sqrt((md.radiusOfCurv**2 - S**2)), -1e3)
    else:
        return -1e5


# In[80]:


swarm = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
layout = uw.swarm.layouts.PerCellRandomLayout(swarm=swarm, particlesPerCell=int(md.ppc))
swarm.populate_using_layout( layout=layout ) # Now use it to populate.
proxyTempVariable = swarm.add_variable( dataType="double", count=1 )
weakMatVariable      = swarm.add_variable( dataType="int", count=1 )
signedDistanceVariable = swarm.add_variable( dataType="double", count=1 )

#
proxyTempVariable.data[:] = 1.0
weakMatVariable.data[:] = 0
signedDistanceVariable.data[:] = 0.0


# In[81]:


#All of these wil be needed by the slab / fault setup functions
#We have two main options, bind them to the TectModel class. 
#or provide them to the functions
#collection them in a dictionary may be a useful way too proviede them to the function 
#wthout blowing out the function arguments

tmUwMap = tm_uw_map([], velocityField, swarm, 
                    signedDistanceVariable, proxyTempVariable, proximityVariable)




# In[94]:


#define fault particle spacing, here ~5 paricles per element
ds = (tm.maxX - tm.minX)/(2.*tm.mesh.elementRes[0])

#we will build a set of interface2D object to allow us to set the initial weak material distribution
fCollection = interface_collection([])

for e in tm.undirected.edges():
    if tm.is_subduction_boundary(e):
        build_slab_distance(tm, e, circGradientFn, md.slabInitMaxDepth, tmUwMap)        
        fb = build_fault(tm, e, circGradientFn, md.faultThickness , md.slabInitMaxDepth, ds, md.faultThickness, tmUwMap)
        fCollection.append(fb)

#
build_slab_temp(tmUwMap, ndp.potentialTemp_, md.slabAge)
fnJointTemp = fn.misc.min(proxyTempVariable,plateTempProxFn)

#And now reevaluate this guy on the swarm
proxyTempVariable.data[:] = fnJointTemp.evaluate(swarm)


# ## Set Weak Layer
# 
# 

# In[98]:


weakMatVariable.data[:] = 0


# In[99]:


for f in fCollection:
    f.rebuild()
    f.set_proximity_director(swarm, weakMatVariable, searchFac = 2., locFac=1.0)


# In[104]:


#figProx = glucifer.Figure(figsize=(960,300) )
#figProx.append( glucifer.objects.Points(swarm , weakMatVariable))
#for f in fCollection:
#    figProx.append( glucifer.objects.Points(f.swarm, pointSize=5))
#figProx.show()


# ## Boundary conditions

# In[109]:


iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]

tWalls = mesh.specialSets["MaxJ_VertexSet"]
bWalls = mesh.specialSets["MinJ_VertexSet"]

#if periodic, we're going to fix a node at the base. 
velnodeset = mesh.specialSets["Empty"]
if md.periodic:
    lWalls = mesh.specialSets["MinI_VertexSet"]
    fixNode = tWalls & lWalls 
    velnodeset += fixNode


velBC  = uw.conditions.DirichletCondition( variable        = velocityField, 
                                           indexSetsPerDof = (iWalls + velnodeset, jWalls) )


# ## Bouyancy

# In[111]:


# Now create a buoyancy force vector using the density and the vertical unit vector. 
thermalDensityFn = rayleighNumber*(1. - proxyTempVariable)

gravity = ( 0.0, -1.0 )

buoyancyMapFn = thermalDensityFn*gravity


# ## Rheology

# In[113]:


symStrainrate = fn.tensor.symmetric( 
                            velocityField.fn_gradient )

#Set up any functions required by the rheology
strainRate_2ndInvariant = fn.tensor.second_invariant( 
                            fn.tensor.symmetric( 
                            velocityField.fn_gradient ))



def safe_visc(func, viscmin=md.viscosityMin, viscmax=md.viscosityMax):
    return fn.misc.max(viscmin, fn.misc.min(viscmax, func))


# In[117]:


temperatureFn = proxyTempVariable


adiabaticCorrectFn = depthFn*ndp.adiabaticTempGrad
dynamicPressureProxyDepthFn = pressureField/pressureDepthGrad
druckerDepthFn = fn.misc.max(0.0, depthFn + md.druckerAlpha*(dynamicPressureProxyDepthFn))

#Diffusion Creep
diffusionUM = (1./ndp.diffusionPreExp)*    fn.math.exp( ((ndp.diffusionEnergyDepth +                    (depthFn*ndp.diffusionVolumeDepth))/((temperatureFn+ adiabaticCorrectFn + ndp.surfaceTemp))))

diffusionUM =     safe_visc(diffusionUM)
    
diffusionLM = ndp.lowerMantleViscFac*(1./ndp.diffusionPreExp)*    fn.math.exp( ((ndp.diffusionEnergyDepth +                    (depthFn*ndp.diffusionVolumeDepth))/((temperatureFn+ adiabaticCorrectFn + ndp.surfaceTemp))))

#diffusionLM =     safe_visc(diffusionLM)


transitionZoneTaperFn = cosine_taper(depthFn, md.lowerMantleDepth - 0.5*md.lowerMantleTransWidth , md.lowerMantleTransWidth )


mantleCreep = diffusionUM*(1. - transitionZoneTaperFn) + transitionZoneTaperFn*diffusionLM

#Define the mantle Plasticity
ys =  ndp.cohesionMantle + (druckerDepthFn*ndp.frictionMantleDepth)
ysf = fn.misc.min(ys, ndp.yieldStressMax)
yielding = ysf/(2.*(strainRate_2ndInvariant) + 1e-15) 

mantleRheologyFn =  safe_visc(mantleCreep*yielding/(mantleCreep + yielding), 
                              viscmin=md.viscosityMin, viscmax=md.viscosityMax)

#Subduction interface viscosity
interfaceViscosityFn = fn.misc.constant(ndp.viscosityFault)


# In[118]:


#create a mapping dictionary that points weakMatVariable variable to fault/interface rheology


viscMapDict = {}
viscMapDict[0] = mantleRheologyFn
for f in fCollection:
    viscMapDict[f.ID] = interfaceViscosityFn
viscMapDict

viscosityMapFn = fn.branching.map( fn_key = weakMatVariable,
                             mapping = viscMapDict)


# ## Stokes

# In[119]:


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


# In[120]:


stokes = uw.systems.Stokes( velocityField  = velocityField, 
                                   pressureField  = pressureField,
                                   conditions     = [velBC,],
                                   fn_viscosity   = viscosityMapFn, 
                                   fn_bodyforce   = buoyancyMapFn )


# In[121]:


solver = uw.systems.Solver(stokes)

solver.set_inner_method("mumps")
solver.options.scr.ksp_type="cg"
solver.set_penalty(1.0e7)
solver.options.scr.ksp_rtol = 1.0e-4


# In[122]:


solver.solve(nonLinearIterate=True, nonLinearTolerance=md.nltol, callback_post_solve = pressure_calibrate)
solver.print_stats()


# ## Swarm Advector

# In[123]:


advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=velocityField, order=2 )


# In[124]:


population_control = uw.swarm.PopulationControl(swarm, deleteThreshold=0.006, 
                                                splitThreshold=0.25,maxDeletions=1, maxSplits=3, aggressive=True,
                                                aggressiveThreshold=0.9, particlesPerCell=int(md.ppc))


# ## Update functions

# In[125]:


# Here we'll handle everything that should be advected - i.e. every timestep
def advect_update():
    # Retrieve the maximum possible timestep for the advection system.
    dt = advector.get_max_dt()
    # Advect swarm
    advector.integrate(dt)

        
    return dt, time+dt, step+1


# In[52]:


#update_faults()


# In[127]:


def update_swarm():
    
    population_control.repopulate()

        
    #A simple depth cutoff for the weakMatVariable
    depthMask = swarm.particleCoordinates.data[:,1] < (1. - md.faultDestroyDepth)
    weakMatVariable.data[depthMask] = 0
    
    
    


# ## Viz

# In[128]:


outputPath = os.path.join(os.path.abspath("."),"output/")

if uw.rank()==0:
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
uw.barrier()


# In[130]:


store1 = glucifer.Store('output/subduction1')
store2 = glucifer.Store('output/subduction2')
#store3 = glucifer.Store('output/subduction3')


figProx = glucifer.Figure(store1, figsize=(960,300) )
figProx.append( glucifer.objects.Points(swarm , weakMatVariable))
for f in fCollection:
    figProx.append( glucifer.objects.Points(f.swarm, pointSize=5))
#figProx.show()

figVisc = glucifer.Figure( store2, figsize=(960,300) )
figVisc.append( glucifer.objects.Points(swarm, viscosityMapFn, pointSize=2, logScale=True) )



#figMask = glucifer.Figure( store3, figsize=(960,300) )
#figMask.append( glucifer.objects.Surface(mesh, pIdFn , valueRange=[0,3]) )
#figMask.append( glucifer.objects.Surface(mesh,  boundMaskFn) )



# ## Main Loop

# In[132]:


time = 0.  # Initial time
step = 0 
maxSteps = 1000      # Maximum timesteps (201 is recommended)
steps_output = 10   # output every 10 timesteps
faults_update = 10
dt_model = 0.
steps_update_model = 10


# In[ ]:


#while time < tg.times[-1]:
while step < maxSteps:
    # Solve non linear Stokes system
    solver.solve(nonLinearIterate=True)
    
    #advect swarm and faults
    dt, time, step =  advect_update()
    dt_model += dt
    
    
    #running fault healing/addition, map back to swarm
    if step % faults_update == 0:
        update_swarm()
        
    
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

