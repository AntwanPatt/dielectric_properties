#!/bin/bash

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


