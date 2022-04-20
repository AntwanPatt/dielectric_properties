import numpy as np

class cryst2lammps:

    def __init__(self, crystfile=None):
        self.infile = crystfile

    def set_base(self, edges=None, angles=90.):
        ## form base matrix A
        ## converting crystallographic vectors in cartesian coordinates
        ## using a formula from: (got at Chemistry StackExchange)
        
        n2 = (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma)
        factz = np.sqrt(np.sin(beta)**2. - n2**2)

        a1 = np.array([a,                 0.,                0.       ])
        a2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.       ])
        a3 = np.array([c * np.cos(beta),  c * n2,            c * facz ])

        A = np.array([[a,                 0.,                0.        ],
                      [b * np.cos(gamma), b * np.sin(gamma), 0.        ],
                      [c * np.cos(beta),  c * n2,            c * factz ]])


    def read(self):
        ## read initial file for the specified format
        ## cartesian or direct coordinates in the file?

    def replicate(self):
        ## multiply it
        ## condition: if bonds already set up can't replicate

    def bonds(self):
        ## finding bonds
