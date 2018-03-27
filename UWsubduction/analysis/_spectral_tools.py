import numpy as np
import underworld as uw
from underworld import function as fn


def spectral_integral(mesh, fnToInt, N, axisIndex=0, kernelFn=1., average = True, 
                      integrationType="volume",surfaceIndexSet=None, returnCoeffs=False ):
    """
    Returns a function that represents the spectral integral over one of the UW mesh axes. 
    The axis over which the integration is performed is parallel to the provided axisIndex.
    (the sin & cosine modes are created orthogonal the axisIndex).
    
    Parameters
    ----------
    mesh: uw.mesh.FeMesh
        The mesh over which integration is performed.
    fnToInt: uw.function.Function
        Function to be integrated.
    N: int
       Total number of modes to use (inluding mode 0)
    axisIndex: int (0, or 1)
       Index of the mesh axis to integrate over
    kernelFn: uw.function.Function
        An additional scalar of vector valued function can be provided. 
        For instance, the upward continuation kernel used in potential field reconstuction
    average: Bool
        If True, include the average (mode 0) component in the returned Fn
    integrationType : str
        Type of integration to perform.  Options are "volume" or "surface".
    surfaceIndexSet : uw.mesh.FeMesh_IndexSet
        Must be provided where integrationType is "surface".
    returnCoeffs: Bool
        If True, return a list of coefficient a_k, b_k in addition to the reconstructed function
   
    
    Notes
    ----------
    In the fourier synthesis, a factor of 2./W needs to be applied to all modes > 0. W is the total width of the spatial domain
    (the axis over which Fourier coeffients were generated). A factor of 1./W needs to be applied to the average/DC component. 
    For more details on these normalizations see http://mathworld.wolfram.com/FourierSeries.html
    
    """
    
    if integrationType:
        if not isinstance( integrationType, str ):
            raise TypeError( "'integrationType' provided must be a string.")
        integrationType = integrationType.lower()
        if integrationType not in ["volume", "surface"]:
            raise ValueError( "'integrationType' string provided must be either 'volume' or 'surface'.")
    if integrationType == "surface":
        if not surfaceIndexSet:
                    raise RuntimeError("For surface integration, you must provide a 'surfaceIndexSet'.")
    
    if axisIndex == 0:
        modeaxes = 1
    elif axisIndex == 1:
        modeaxes = 0
    else:
        raise ValueError( "axisIndex must either of 0 or 1")
    
    if N <=2:
        raise ValueError( "N must be at least 2, otherwise you should use an integral (N=1)")
        
    # create set of wavenumbers / modes
    res = mesh.elementRes[modeaxes]
    width = abs(mesh.maxCoord[modeaxes] - mesh.minCoord[modeaxes])
    height = abs(mesh.maxCoord[axisIndex] - mesh.minCoord[axisIndex])
    ax_ = fn.coord()[modeaxes]
    modes = []
    #ks = []
    for i in range(1,N):
        factor = float(i)*2.*np.pi/width
        modes.append(factor*ax_)
        #ks.append(factor)
        
    sinfns = fn.math.sin(modes)
    cosfns = fn.math.cos(modes)
    
    
    
    if average:
        #average_ = uw.utils.Integral((2./width)*fnToInt,mesh)
        average_ = uw.utils.Integral(fnToInt,mesh)

    else:
        average_ = fn.misc.constant(0.)
    
    if integrationType=="volume":
        sin_coeffs = uw.utils.Integral(fnToInt*kernelFn*sinfns,mesh)
        cos_coeffs = uw.utils.Integral(fnToInt*kernelFn*cosfns,mesh)

    else:
        sin_coeffs = uw.utils.Integral(fnToInt*kernelFn*sinfns, mesh, 
                                        integrationType='surface', 
                                       surfaceIndexSet=surfaceIndexSet)
        cos_coeffs = uw.utils.Integral(fnToInt*kernelFn*cosfns, mesh, 
                                        integrationType='surface', 
                                       surfaceIndexSet=surfaceIndexSet) 

    synthFn = (2./width)*fn.math.dot(sin_coeffs.evaluate(),sinfns) + \
              (2./width)*fn.math.dot(cos_coeffs.evaluate(),cosfns) + \
                         (1./width)*average_.evaluate()[0]
    
    if not returnCoeffs:
        return synthFn
    else:
        return synthFn, [sin_coeffs, cos_coeffs]
    
    
def integral_wavenumbers(mesh, N, axisIndex=0):
    
    """
    Returns a set of N - 1  angular wavenumbers (2.*pi*N/width)
    width is the length of the mesh along the axis given by axisIndex
    N takes the values (1, ..., N - 1), (the zeroth wavenumber is is not given) 
    
    Parameters
    ----------
    mesh: uw.mesh.FeMesh
        The mesh over which integration is performed.
    N: int
       Total number of modes to use  in the spectral integral (it is assumed that this includes mode 0)
    axisIndex: int (0, or 1)
       Index of the mesh axis to integrate over
    
    """
    
    if axisIndex == 0:
        modeaxes = 1
    elif axisIndex == 1:
        modeaxes = 0
    else:
        raise ValueError( "axisIndex must either of 0 or 1")
        
    if N <=2:
        return np.array([0])
    
    #res = mesh.elementRes[modeaxes]
    width = mesh.maxCoord[modeaxes] - mesh.minCoord[modeaxes]

    ks = []
    for i in range(1,N):
        factor = float(i)*2.*np.pi/width
        ks.append(factor)
    return(np.array(ks))