# UWSubduction




## Overview

The UWsubduction module provides tools related to thermo-mechanical subduction simulation in Underworld2.

The code is designed to facilitate simple development of 2D parallel thermo-mechanical subduction experiments. 

Here you will find the the source code, examples designed to get you started, and a place to post issues when things don't work.

## Getting started

The `.\Background` directory contains a series of Jupyter notebooks that introduce some of the key objects (Classes) that we have implemented, and are used in the Examples.

The `.\Examples` directory contains a simple subduction model (as Jupyter notebooks). While notebooks can be run in serial at low resolution, the models are intended to be run in parallel, which is easiest to achieve by converting the notebook to a script, e.g:

```
jupyter nbconvert --to script 2d_subduction.ipynb
```



## Installation

To use UWsubduction you will need a working copy of Underworld2 (python3 version). In addition, the UWsubduction module requires the following python packages (which can be pip installed):

* networkx
* easydict
* naturalsort
* networkx (version 1.11)
* pint (now a UW2 requirement also)



### Docker

A docker script that builds all requirements is found in the `UWsubduction/Docker` folder

A prebuilt Docker image is avalaible here: `dansand/uwsubduction`


To run the 2d_subduction.py UWsubduction model on NPROCS (an integer) using the docker image you can type:

```
docker run -v $PWD:/workspace dansand/uwsubduction mpirun -np NPROCS 2d_subduction.py [command line args]

```


## Publications

Sandiford, Dan, and Louis Moresi. "Improving subduction interface implementation in dynamic numerical models." Solid Earth 10.3 (2019): 969-985.

Sandiford, Dan, et al. "Geometric controls on flat slab seismicity." Earth and Planetary Science Letters 527 (2019): 115787.

Sandiford, Dan, et al. "The fingerprints of flexure in slab seismicity." (2019).

## License

This code was written by Dan Sandiford, Louis Moresi and the Underworld Team. It is licensed under the Creative Commons Attribution 4.0 International License . We offer this licence to encourage you to modify and share the examples and use them to help you in your research.


<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />
<hr>
