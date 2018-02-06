# graph Updates

*Migration of a ridge in the above type of model can be simulated simply by specifying appropriate plate velocities and a velocity of the boundary at which the plates meet which is the average of the plate velocities*

Davies, 1986

this implies symmetric spreading. 



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



