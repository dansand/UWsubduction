# Notes

## Models

### Bugs

* Stokes callbacks aren't actually implemented
* buoyancy factor
* in ex1 ex2 the horizontal resolution is still wrong
* ex2 is not properly safe yet - need to ran at 24 procs to catch the bug (need two rows of elements )
* 1400 degree potential temp
* Im using the constant 1. for the dimensionless diffusivity in the plate and slab creation routines. Kind of opaque.
* make the viscosityMapFns smart - need to map all proximities to interfac rheology
* dampen the strain-rate-depndent plate boundary calls
  ​

## TectModel

### Design

all methods taking a plate pair should take a tuple of plate Ids

### Bugs

## Marker2D

### Design

### Bugs

## Utilities

### Design 

### Bugs

* circGradientFunction doesn't work?



