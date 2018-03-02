##

"""
analysis tools.
"""
import numpy as np

def eig2d(sigma):

    """
    Input: sigma, symmetric tensor, numpy array of length 3, with xx yy xy compenents
    Output:
    s1: first major stress eigenvector will be the most extensive
    s2: second major stress the least extensive, most compressive
    deg: angle to the first major stress axis (most extensive in degrees anticlockwise from horizontal axis - x)
    """


    s11=sigma[0]
    #s12=sigma[2]/2.  #(engineering strain/stress)
    s12=sigma[2]
    s22=sigma[1]

    fac = 28.64788975654116; #90/pi - 2theta conversion

    x1 = (s11 + s22)/2.0;
    x2 = (s11 - s22)/2.0;
    R = x2 * x2 + s12 * s12;

    #Get the stresses
    if(R > 0.0):         #if shear stress is not zero
        R = np.sqrt(R);
        s1 = x1 + R;
        s2 = x1 - R;
    else:
        s1 = x1;        #if shear stress is zero
        s2 = x1;

    if(x2 != 0.0):
        deg = fac * np.arctan2(s12,x2); #Return the arc tangent (measured in radians) of y/x. The fac contains the 2-theta
    #else, the normal componets are equa - we are at the angle of maximum shear stress
    elif s12 <= 0.0:
        deg= -45.0;
    else:
        deg=  45.0;
    return s1, s2, deg
