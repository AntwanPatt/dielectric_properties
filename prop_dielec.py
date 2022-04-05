#!/usr/bin/python

""" This script makes use of the total dipole moment (M) statistical
    distribution to compute the dielectric constant of the system.
    The distribution of M is obtained by molecular dynamics simulations. Only
    LAMMPS at the moment.
"""
import numpy as np
import numpy.linalg as nl

## some constants
eps_0 = 8.85418781762039e-12 # vacuum permittivity
kB    = 1.380649e-23         # boltzmann constant
e2C   = 1.602176634e-19      # elementary charge in C
D2eA  = 0.20819434           # debye to e.A
fac   = 1.112650021e-59      # conversion factor Debye^2->C^2m^2

class prop_dielec:
    """Pyton class to compute dielectric properties"""

    def __init__(self, tot_dip_moments, arg2):
        # getting total dipole moments distribution and converting it to Debye
        self.M = tot_dip_moments / D2eA

        # time between two estimates of M
        self.dt = None

        self.truc = arg2

    def static_eps(self):
        ## initializing time vector
        time = np.arange(len(self.M)) * self.dt
        
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

        ## need temperature and volume to calculate
        ## TO DO: * insert a thermo analysis class here
        eps_r = 0.0
        thermo_file = "avg.res"
        try:
            temperature = np.mean(np.loadtxt(thermo_file, skiprows=2, usecols=(1)))
            volume      = np.mean(np.loadtxt(thermo_file, skiprows=2, usecols=(4)))
        except FileNotFoundError:
            print("The file containing the thermodynamic quantities, {:s}, \
                    \ cannot be found".format(thermo_file))
            exit()

        ## calculating static dielectric constant (relative permittivity), eps_r
        eps_r = 1 + (Mdiff * fac) / (3 * eps_0 * volume * 1e-30 * kB * temperature)

        return eps_r

################################################################################

# test code of the above class
if __name__ == '__main__':
    A = np.ones((3,3))
    test = prop_dielec(A, "haha")
