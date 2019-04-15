from easydict import EasyDict as edict

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
