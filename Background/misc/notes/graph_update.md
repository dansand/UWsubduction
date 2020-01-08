# graph Updates

## Overview

The aim is to provide a number of methods fot tracking the migration of plate boundaries, which can be composed to form a graph update function. For fullydyanmic models, one might have

```python
def graphUpdateFn(tectModel, time=0.0, dt = 0.0):
    for e in tectModel.edges():
        #boundary / ficticious edge
        if e[0] == e[1]:
            pass
        #internal ridges or subdction zones
        else:
          newx = strainRateMinFn(maskFn)
          tectModel.set_bound_loc(e, newx) 
```

Alternatively, if some subduction boundary velocities are provided 

```python
def graphUpdateFn(tectModel, time=0.0, dt = 0.0):
    for e in tectModel.edges():
        #idx_ = [tectModel.get_index_at_time(time)]
        if tectModel.egdes(e)['vels'][idx_]:
        #   dx = tectModel.egdes(e)['vels'][idx_] * dt
        #   x= tectModel.get_bound_loc(e) 
        #   tectModel.set_bound_loc(e, x + newx)
            newx = boundVelUpdateFn(tectModel,e, time, dt)
            tectModel.set_bound_loc(e, x) 
        #boundary / ficticious edge
        elif e[0] == e[1]:
            pass
        #internal ridges or subdction zones
        else:
          newx = strainRateMinFn(tectModel,e, maskFn)
          tectModel.set_bound_loc(e, newx) 
```



## A list of udpate Functions

* `boundVelUpdateFn(tectModel, e, time, dt)`
  * where a plate boundary velocity is specified
* `strainRateMinFn(tectModel, e, maskFn)`
  * where no plate or boundary vels are specified
* `subductionUpperPlateFn(tectModel, e, dt)`
  * where an SZ upper plate vel is sepcified
* `symmetricRidgeUpdateFn(tectModel, e)`
  * where one or both plate Vels are specified
* `boundary_ridge_migration(tectModel,e )`:
  * boundary ridges can be made to follow their plate (they are defined by a plate ID. )

## Additional functions 

* `is_ridge()`
* `is_self_loop()`


* `set_bound_loc(e, newx) `
* `get_bound_loc(e)`
* `get_bound_vel(e, time)`
* `get_upper_plate_vel()`
* `globalVelMinMix(e, dist)`



## Mask functions

* `plate_boundary_mask_fn`
* `ridge_mask_fn`
* `subduction_mask_fn`
* `combine_mask_fn`
* `interior_plate_mask_function(self, lambda, minPlateLength)`



*Migration of a ridge in the above type of model can be simulated simply by specifying appropriate plate velocities and a velocity of the boundary at which the plates meet which is the average of the plate velocities*

Davies, 1986

this implies symmetric spreading. 



There are at least 4 cases where I want to mask the plates /ridges

- for velocity boundary condtions

  - these are likely to be symmetric, but may have a dependence on plate size

- for fault location

  - these should have a dependence on plate size, and may not be symmetric

- for plate velocity evaluation - i.e. intenal plate vel. average / max / min

  - these are likely to be symmetric, but may have a dependence on plate size

- for strain rate evaluation, both for finding exiting bounary locations. and finding new plate boundaries

  - these are likely to be symmetric, but may have a dependence on plate size

  ​



​		
​	

## Fully dyanmic models

In fully dyamic models plate boundaries can be updated by tracking the velocities in the region proximal to the current location. 

## Fully constrained models

When the surface velocities of the plates is set, we meet an asymmetry. Subduction zones can be updated with reasonable accuracy, based on plate kinematics alone. IN the absene of signical upper plate deformation, the trench migration velocity is similar ot the upper plate velocity,. 

On the other hand, the evolution of midocean ridges cannot be determined from the `averge' velocities of the asjacent plates. This is becase the symmetry of the spreading is not known, unless we look at the velocity field in fine detail. 

If divergent flow is specified, the ridge will remain fixed in space unless we specify a  a ridge evolution pathway. 

## Partially constrained models

If the kinematocs of one conjugate plate are specified, we have two options: let the ridge update dyanamically, or specify a ridge evolution pathway. 



## Updatng the graph

Basically consiste of the follwoifn decisions:

### SZs

*  for SZ, an upper plate velouty guess can be made or a strain rate appraoch. 
*  If SZ, and upper plate or SZ velocities are specified, can make a kinematic guess. 

### RiDges

* if no plate velicties of bounary velocities are specified, use a strain rate apprach for Ridges (find the local strain rate minima), 
* If both plate velociteis are specifeid, bu no ridge velocity is specified, the ridge remains static, alternitively, under the symmetric spreading scenarion, the ridge can be updated 
* If one one plate velocity is specified. The ridge is likely but not guaranteed to rmain in the same plate; a strain rate apprach should be used. 

## Constraints / Rules 

* If ridges have self loops, their location should not be updated. 
* If ridge velocity is greater than plate velocity it implies ridge jump. 
* if (X_r - X_sz) < tol, it implies ridge subduction.  In this case, a node in the graph should be disconected
* If new plates are added, there is a hierarchy in terms of how to reorder the graph. 



