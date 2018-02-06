# UWsubduction

## Contents

* introduction
* Challenges in Subduction modelling
* code - main Classes
* installation



## Introduction

This sub-repository (UWsubduction) hosts code specifically related to the simulation of thermo-mechanical subduction within the broader Underworld2 ecosystem (do we still say framework?). This code was developed by Dan Sandiford and Louis Moresi at the University of Melbourne. 

The code is designed to facilitate simple develpment of 2D thermo-mechanical subduction experiments, using a range of objects that we have develped. Although aimed at 2D conection, obtsainign reasoable results (in a matter of less than days) will require running models in parallel. 

## Challenges in Subduction modelling


Subduction consists of sinking of ~ 100 km thick lithospheric sheets (cold plumes), and passive mantle counterflow on the order of ~ 1000s km. Yet these convective motions are also controlled by ~ km-thick faults (the subduction interface). Historically, the primary chanllenges (CFD) models of subduction has been managing the significant (extreme ?) contrast in length scale, as well as solving Stokes' equations for very large viscosity contrasts. A further problem involves the emergence of these faults. 

In many cases, subduction problems can be studied while relaxing some of these contrainst. For instance, spatial and temporal averging at the level of the megathrust appears to be reasonable. Additionally,it is comonplace to impose a pre-existing fault, along with some process for it's evolution, without worring about fault emergence. A very common approach is to apply a thin layer of weak material on top of sbducting plates, which enables a process of self-lubrication, or entrainment. Often this involves pre-emting the general dyamics of the flow -what we might term the topological structure of flow. 


We also want to able to use a mixture of boundary conditions, allowing us to impose the kinematics of some parts of the model, while allowing other parts to evolve fully-dynamically. Because faults (i.e. weak crust) are geneally imposed structures, there needs to be some level of communication between the evolving tecteonic model (i.e. number and location of plates) and the faults that enable continued (one-sided) subduction. 



## Main Classes

### markerLine2D

### TectModel

The TectModel class is abstraction of a 2D plate-tectonic system, based on a (directed) graph. In fact, we build directly on top of the Python networkx DiGraph object. 




