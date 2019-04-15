#try:
#    import unsupported.scaling as sub_scaling;
#except:
#    import unsupported.geodynamics.scaling as sub_scaling;

from underworld import scaling as sub_scaling


from minimal_example import pd, md
from easydict import EasyDict as edict

paramDict_dim = pd
modelDict_dim = md

#####################
#Next, define a standard set of scale factors used to non-dimensionalize the system
#####################


KL = pd.refLength
KT = pd.potentialTemp - pd.surfaceTemp
Kt = KL**2/pd.refDiffusivity
KM = pd.refViscosity * KL * Kt

scaling_coefficients = sub_scaling.get_coefficients()

scaling_coefficients["[length]"]      = KL.to_base_units()
scaling_coefficients["[temperature]"] = KT.to_base_units()
scaling_coefficients["[mass]"]        = KM.to_base_units()
scaling_coefficients["[time]"] =        Kt.to_base_units()


#####################
#Now we map pd, md to non-nonDimensionalized dictionaries, paramDict, modelDict
#####################

def build_nondim_dict(d, sca):
    ndd = edict({})
    for key, val in d.items():
        #can only call .magnitude on Pint quantities
        if hasattr(val, 'dimensionality'):
            if val.unitless:
                ndd[key] = val.magnitude
            else:
                ndd[key] = sca.nonDimensionalize(val)

        else:
            ndd[key] = val

    return ndd


#build the dimensionless dictionaries
paramDict  = build_nondim_dict(pd, sub_scaling)
modelDict= build_nondim_dict(md, sub_scaling)



#####################
#Finally, define some dimensional numbers and scaling factors
#####################
