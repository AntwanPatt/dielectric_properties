#!/usr/bin/python

""" This script makes use of the total dipole moment
    statistical distribution to compute the dielectric
    constant of the system.
"""
import numpy as np
import numpy.linalg as nl

## some constants
eps0 = 8.85418781762039e-12 # vacuum permittivity
kb   = 1.380649e-23         # boltzmann constant
e2C  = 1.602176634e-19      # elementary charge in C
D2eA = 0.20819434           # debye to e.A
fac  = 1.112650021e-59      # conversion factor Debye^2->C^2m^2

class prop_dielec:
    """Pyton class to compute dielectric properties"""

    def __init__(self, tot_dip_moments, arg2):
        self.M = tot_dip_moments
        self.truc = arg2

    def static_eps(self):
        ## initializing vectors for <|M|^2> and |<M>|^2
        M2_avg = np.zeros((len(self.M), 1))
        Mavg_2 = np.zeros((len(self.M), 1))

        ## running averages
        for i in range(len(self.M)):
            Mtemp = self.M[:i+1,:]
            M2_avg[i] = np.mean( nl.norm(Mtemp, axis = 1) ** 2 )
            Mavg_2[i] = nl.norm( np.mean(Mtemp, axis = 0) ) ** 2

            ## calculating variance and M ratio for convergence
            Mdiff  = M2_avg - Mavg_2
            Mratio = np.divide(Mavg_2, M2_avg)

            ## calculating static dielectric constant (relative permittivity)
            temperature = np.mean(np.loadtxt("avg.res", skiprows=2, usecols=(1)))
            volume      = np.mean(np.loadtxt("avg.res", skiprows=2, usecols=(4)))

            eps = 1 + (Mdiff * fac) / (3 * eps0 * volume * 1e-30 * kb * temperature )

################################################################################

# test code of the above class
if __name__ == '__main__':
    A = np.ones(3)
    test = prop_dielec(A, "haha")
