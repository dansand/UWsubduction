from easydict import EasyDict as edict
import underworld as uw
from underworld import function as fn


class phases():
    """
    Class that allows you to create 'phase functions' for a mineral component

    """
    def __init__(self,name, depths,temps, widths,claps,buoyancies):
        """
        Class initialiser.

        Parameter
        ---------
        name : str
            'Component', e.g olivine, pyroxene-garnet
        depths: list
            list of transition depths in model units
        temps: list
            list of transition depths in model units
            Typical scaling will be temps(') = (temps(K) - tempSurf(K))/deltaT(K)
        widths: list
            list of transition temps (temp along adiabat) model units
        claps: list
            list of Clapeyron slopes in model units
            Typical scaling will be claps(') = claps(Pa/K)*slopeFactor
            slopeFactor = (tempscale/(densityscale*gravityscale*lengthscale)
        buoyancies: list
            list of density changes associeated with each phase transition in model units.
            Density change is considered positive if density increseas as depth increases,
            i.e. downward-traversing the phase transition
            Typical scaling will be buoyancies(') = densities(kg/m3)*buoyancyFactor
            buoyancyFactor = (gravityScale*lengthScale**3)/(viscosityScale*diffusivityScale)

        Returns
        -------
        mesh : ndp
        Dictionary storing the phase-transition values (ndp referring to non-dimensional parameter)

        """
        if not len(depths) == len(widths) == len(claps) == len(buoyancies):
            raise ValueError( "All lists of phase values should be the same length")
        self.ndp = edict({})
        self.ndp.name = name
        self.ndp.depths = depths
        self.ndp.temps = temps
        self.ndp.widths = widths
        self.ndp.claps = claps
        self.ndp.buoyancies = buoyancies

    def nd_reduced_pressure(self, depthFn, temperatureField, depthPh, clapPh, tempPh):
        """
        Creates an Underworld function, representing the 'reduced pressure'
        """

        return (depthFn - depthPh) - clapPh*(temperatureField - tempPh)

    def nd_phase(self, reduced_p, widthPh):
        """
        Creates an Underworld function, representing the phase function in the domain
        """
        return 0.5*(1. + fn.math.tanh(reduced_p/(widthPh)))

    def phase_function_sum(self, temperatureField, depthFn):
        """
        Creates an Underworld function, representing the Sum of the individual phase functions,
        mainly for testing implementation

        Parameter
        ---------
        temperatureField : underworld.mesh._meshvariable.MeshVariable
            mesh variable holding the model temperature field
        depthFn : underworld.function._function
            function representing depth from surface (model dimensions)
        """

        pf_sum = uw.function.misc.constant(0.)

        for phaseId in range(len(self.dp['depths'])):
            #build reduced pressure
            rp = self.nd_reduced_pressure(depthFn,
                                   temperatureField,
                                   self.ndp['depths'][phaseId ],
                                   self.ndp['claps'][phaseId ],
                                   self.ndp['temps'][phaseId ])
            #build phase function
            pf = self.nd_phase(rp, self.ndp['widths'][phaseId ])
            pf_sum += pf

        return pf_sum

    def buoyancy_sum(self, temperatureField, depthFn):
        """
        Creates an Underworld function, representing the Sum of the individual phase functions...
        and the associated density changes:

        pf_sum = Sum_k{ (Ra*delRho_k*pf_k/rho_0*eta_0*delta_t)}

        Parameter
        ---------
        temperatureField : underworld.mesh._meshvariable.MeshVariable
            mesh variable holding the model temperature field
        depthFn : underworld.function._function
            function representing depth from surface (model dimensions)


        """
        #bouyancyFactor = (gravityscale*lengthscale**3)/(viscosityscale*diffusivityscale)

        pf_sum = uw.function.misc.constant(0.)

        for phaseId in range(len(self.ndp['depths'])):
            #build reduced pressure
            rp = self.nd_reduced_pressure(depthFn,
                                   temperatureField,
                                   self.ndp['depths'][phaseId ],
                                   self.ndp['claps'][phaseId ],
                                   self.ndp['temps'][phaseId ])
            #build phase function
            pf = self.nd_phase(rp, self.ndp['widths'][phaseId ])
            pf_sum += pf*self.ndp['buoyancies'][phaseId ] #we want the dimensional buoyancies here

        return pf_sum
