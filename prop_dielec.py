#!/usr/bin/python

""" This script makes use of the total dipole moment (M) statistical
    distribution to compute the dielectric constant of the system.
    The distribution of M is obtained by molecular dynamics simulations. Only
    LAMMPS at the moment.
"""
import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import trapz
import scipy.spatial as sci

## some constants
eps_0 = 8.85418781762039e-12 # vacuum permittivity
kB    = 1.380649e-23         # boltzmann constant
e2C   = 1.602176634e-19      # elementary charge in C
D2eA  = 0.20819434           # debye to e.A
A2m   = 1.0e-30              # angstrom to meter
fac   = 1.112650021e-59      # conversion factor Debye^2->C^2m^2

################################################################################
class prop_dielec:
    """Pyton class to compute dielectric properties"""

    def __init__(self, Mfrom = 'res', Mfile = None, dt = None):
        if Mfrom not in ['res', 'traj']:
            raise ValueError("Total dipole moments distribution has to be obtained \
either directly from results file, Mfrom = 'res', or by analysing \
trajectory file, Mfrom = 'traj'.")
        else:
            self.Mfrom = Mfrom

        if dt is None:
            raise ValueError("The timestep (in fs) used in the simulation is not provided.")
        else:
            self.dt = dt

        if Mfile is None:
            raise FileNotFoundError("Filename missing to obtain total dipole moments distribution.")
        else:
            self.Mfile = Mfile

        self.dtM = None

        self.getM()

    def getM(self):
        # function to get M either reading output file from LAMMPS
        # or by analysing trajectory
        if self.Mfrom == 'res':
            # timesteps number vector
            self.time = np.loadtxt(self.Mfile, usecols=(0)) * self.dt
            # time between two estimates of M
            self.dtM   = (self.time[1]-self.time[0])
            # getting total dipole moments distribution and converting it to Debye
            self.M    = np.loadtxt(self.Mfile, usecols=(1,2,3)) / D2eA

        elif self.Mfrom == 'traj':
            print("This is traj")
            # Opening trajectory file
            f = open(self.Mfile, "r")
            lines = f.readlines()

            # Counting number of configurations
            conf = 0
            for line in lines:
                if "TIMESTEP" in line:
                    conf += 1

            # Setting up some variables
            start_conf = 0

            self.M = np.zeros((conf - start_conf, 3))

            conf_count = 0
            line_count = 0

            # Reading the number of atoms
            Natoms = int(lines[line_count+3])

            ### Processing the configurations
            while (conf_count < (conf - start_conf)):
                #print(conf_count + start_conf)
                # Updating the line count
                line_count = (9 + Natoms) * (conf_count + start_conf)

                # Reading the box length of the configuration
                # !!! Care with handling orthogonal or triclinic cell, not same format in LAMMPS output
                #L = np.zeros(3) + float(lines[line_count+5].split()[1])*2
                if "xy" in lines[line_count + 4]:
                    #lines[line_count+7]
                    xlo, xhi, xy = np.array(lines[line_count+5].split())
                    ylo, yhi, xz = np.array(lines[line_count+6].split())
                    zlo, zhi, yz = np.array(lines[line_count+7].split())
                    lx = xhi - xlo
                    ly = yhi - ylo
                    lz = zhi - zlo

                    a = lx
                    b = np.sqrt(ly**2 + xy**2)
                    c = np.sqrt(lz**2 + xz**2 + yz**2)
                    alpha = np.arccos((xy * xz + ly * yz) / (b * c))
                    beta = np.arccos(xz / c)
                    gamma = np.arccos(xy / b)

                    edges  = np.array([a, b, c])
                    angles = np.array([alpha, beta, gamma])

                else:
                    xlo, xhi = np.array(lines[line_count+5].split()).astype(float)
                    ylo, yhi = np.array(lines[line_count+6].split()).astype(float)
                    zlo, zhi = np.array(lines[line_count+7].split()).astype(float)
                    lx = xhi - xlo
                    ly = yhi - ylo
                    lz = zhi - zlo

                    a = lx
                    b = ly
                    c = lz

                    edges  = np.array([a, b, c])
                    angles = np.array([90.0, 90.0, 90.0])

                A_f2c = A_c2f = np.zeros((3, 3))
                A_f2c, A_c2f = self.coord_conv_matrix(edges, angles)

                # and the rest of the data (label, positions, charge) of each atoms
                data = np.loadtxt(self.Mfile, skiprows=line_count+9, max_rows=Natoms)
                pos  = data[:,2:5]

                ### Calculating total dipole moment for a frame
                whole = True
                # Whole box
                if (whole):
                    # Identifying the positive and negative charge
                    ind_posQ = data[:, 5] > 0.
                    ind_negQ = data[:, 5] < 0.

                    # Centers of charges
                    posQ = data[ind_posQ,5].reshape(-1,1)
                    negQ = data[ind_negQ,5].reshape(-1,1)

                    r_posQ = data[ind_posQ,2:5]
                    self.reconstruct_box(r_posQ, edges)
                    r_negQ = data[ind_negQ,2:5]
                    self.reconstruct_box(r_negQ, edges)

                    print(np.max(r_posQ))
                    print(np.max(r_negQ))

                    #print(r_negQ)
                    #print(negQ)
                    #A = r_negQ * negQ
                    #print(np.sum(A[:,0])/np.sum(negQ))

                    #bar_posQ = np.sum(r_posQ * posQ, axis = 0) / np.sum(posQ)
                    #bar_negQ = np.sum(r_negQ * negQ, axis = 0) / np.sum(negQ)

                    bar_posQ = np.sum(r_posQ, axis = 0) / len(r_posQ)
                    bar_negQ = np.sum(r_negQ, axis = 0) / len(r_negQ)

                    #print(bar_negQ)
                    print(bar_negQ-bar_posQ)
                    print((bar_negQ-bar_posQ)*np.sum(posQ))
                    print(np.sum(posQ))
                    print(np.linalg.norm((bar_negQ-bar_posQ)*np.sum(posQ)))

                    # Check periodic boundary conditions for the distance between
                    # the two charge barycentres

                    # Calculate the total dipole moment of the frame


                    # Summing molecular dipole moments
                    #self.M[conf_count] = np.sum(mu, axis=0)
                    #M[conf_count] = np.linalg.norm(np.sum(mu, axis=0))

                dipoles = False
                if (dipoles):
                    print("dipoles")
                    nSites = 3
                    # Preparing configuration for periodic kdtree
                    self.reconstruct_box(pos, edges)
                    # Computing the tree
                    tree = sci.cKDTree(pos, boxsize=edges)

                    # Getting the indexes of all oxygen atoms
                    O = np.where(data[:,1] == 1)[0]
                    # Searching for the neighbors of O atoms in a given radius
                    # to identify the water molecules
                    O_neigh = tree.query_ball_point(pos[O], r=1.2)

                    # Searching for uncomplete water molecules
                    excluded = np.array([])
                    for t in range(len(O_neigh)):
                        if len(O_neigh[t]) != nSites:
                            excluded=np.append(excluded, t)

                    excluded=excluded.astype(int)

                    # Dipole moment array initialisation
                    mu = np.zeros([len(O)-len(excluded),3])

                    j = 0
                    # Loop on all O atoms
                    for i in range(len(O)):
                        # but just on whole molecules
                        if i in excluded: continue

                        # Identifying the charges in the molecule
                        ind_posQ = np.array(O_neigh[i])[data[O_neigh[i], 5] > 0.]
                        ind_negQ = np.array(O_neigh[i])[data[O_neigh[i], 5] < 0.]
                        # Calculating a mean charge because of non exact local neutrality with reax
                        Qpos = np.sum(data[ind_posQ, 5])
                        Qneg = np.sum(data[ind_negQ, 5])
                        Q    = (Qpos - Qneg) / 2.

                        # Barycentre of charges and taking into account PBC
                        L = edges[0]
                        bar_pos = np.sum((pos[ind_posQ] - pos[O[i]]) - L*np.around((pos[ind_posQ] - pos[O[i]])/L), axis=0)/len(ind_posQ)
                        bar_neg = np.sum((pos[ind_negQ] - pos[O[i]]) - L*np.around((pos[ind_negQ] - pos[O[i]])/L), axis=0)/len(ind_negQ)

                        # Direction vector between the 2 barycentres and PBC
                        pair_bar  = bar_pos - bar_neg
                        pair_bar -= L * np.around(pair_bar / L)

                        # Computing dipole moment for a molecule
                        mu[j] = pair_bar * Q
                        j += 1


                if (dipoles):
                    print(np.sum(mu, axis=0))

                conf_count += 1

    def reconstruct_box(self, positions, box):
        """ Reconstruct the periodic box by
            putting every atom in the box,
            between {[0, L_i]}
        """
        for i in range(0,3):
            ind_neg = positions[:,i] < 0
            ind_pos = positions[:,i] > box[i]
            positions[ind_neg,i] += box[i]
            positions[ind_pos,i] -= box[i]

        return positions

    def coord_conv_matrix(self, edges, angles, angle_in_degrees = True):
        try:
            a, b, c = edges
        except:
            a = b = c = edges

        try:
            alpha, beta, gamma = angles
        except:
            alpha = beta = gamma = angles

        if angle_in_degrees:
            angles = np.deg2rad(angles)

        cosa, cosb, cosg = np.cos(angles)
        sina, sinb, sing = np.sin(angles)

        V = 1.0 - cosa**2.0 - cosb**2.0 - cosg**2.0 + 2.0 * cosa * cosb * cosg
        V = np.sqrt(V)

        # Frac to cart
        n1 = (cosa - cosb * cosg) / sing
        A_f2c = np.zeros((3, 3))
        A_f2c = np.array([[a,  b * cosg, c * cosb     ],
                          [0., b * sing, c * n1       ],
                          [0., 0.,       c * V / sing]])
        A_f2c = np.where(np.abs(A_f2c) < 1e-9, 0.0, A_f2c)

        # Cart to frac
        n2 = (cosa * cosg - cosb) / sing
        n3 = (cosb * cosg - cosa) / sing

        A_c2f = np.zeros((3, 3))
        A_c2f = np.array([[1.0 / a, -cosg / (a * sing), n2   / (a * V)],
                          [0.,      1.0   / (b * sing), n3   / (b * V)],
                          [0.,      0.,                 sing / (c * V)]])
        A_c2f = np.where(np.abs(A_c2f) < 1e-9, 0.0, A_c2f)

        return A_f2c, A_c2f

    def calc_stateps(self):
        ## initializing vectors for <|M|^2> and |<M>|^2
        M2_avg = np.zeros(len(self.M))
        Mavg_2 = np.zeros(len(self.M))

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

        try:
            thermo_file = "avg.res"
            temperature = np.mean(np.loadtxt(thermo_file, skiprows=2, usecols=(1)))
            volume      = np.mean(np.loadtxt(thermo_file, skiprows=2, usecols=(4)))
        except FileNotFoundError:
            print("The file containing the thermodynamic quantities, {:s}, \
                    \ cannot be found".format(thermo_file))
            exit()

        ## calculating static dielectric constant (relative permittivity), eps_r
        eps_r = 1 + (Mdiff * fac) / (3 * eps_0 * volume * A2m * kB * temperature)

        ## outputting the results
        outfile = "Mtot2_epsr.res"
        outres  = np.vstack((self.time, M2_avg, Mavg_2, Mdiff, Mratio, eps_r)).transpose()
        header  = "Running averages of Mtot related quantities and static epsilon"
        header += "\n Time (fs) - <|M|^2> (D^2) - |<M>|^2 (D^2) - <|M|^2>-|<M>|^2 (D^2) - <|M|^2>/|<M>|^2 - epsilon_r"
        np.savetxt(outfile, outres, header = header)

        return eps_r

    def autocorr(self, X):
        """ the convolution is actually being done here
        meaning from -inf to inf so we only want half the
        array"""

        result = np.correlate(X, X, mode='full')
        return result[int(result.size/2):]

    def calc_ACFM(self, cortime = None):
        if cortime is None:
            raise ValueError("The time window (in fs) for the ACF calculation is not provided.")
        else:
            self.cortime = cortime

        # for ACF, must check if dt is sufficiently lower than correlen
        if self.cortime < self.dtM * 1e3:
            raise ValueError("The time window must exceed the time between two values \
            of M by at least a thousand folds, for better statistics.")

        corsteps = int(np.floor(self.cortime / self.dtM))

        self.MACF = np.zeros(corsteps)

        blocks = range(corsteps, len(self.M), corsteps)

        for t in blocks:
            for i in range(3):
                self.MACF += self.autocorr(self.M[t-corsteps:t,i])

        ## normalization of the ACF of M by the numbler of blocks and M(0)**2
        self.MACF /= len(blocks)
        self.MACF /= self.MACF[0]

        ## time vector for MACF in the given time window
        self.tACF = np.arange(corsteps) * self.dtM

        data = np.vstack((self.tACF, self.MACF)).transpose()

        np.savetxt("ACF.res", data)

        #plt.plot(x,MACF,'k')
        #plt.plot(x, y_fitted)
        #plt.ylabel(r'$\frac{M(t)\cdot M(0)}{M(0)\cdot M(0)}$',fontsize=20)
        #plt.xlabel('Time [ps]')
        #plt.show()

    def integ_ACFM(self):
        I_trapz = trapz(self.MACF, self.tACF)
        return I_trapz

    def fit_ACFM(self):
        ## Debye model
        popt, pcov = curve_fit(lambda t, tau: np.exp(- t / tau), self.tACF, self.MACF, p0=(10))

        tau = popt[0]

        y_fitted = np.exp(- self.tACF / tau)
        y_fitted= y_fitted.reshape(len(y_fitted),1)

        return tau


#    def fft_ACFM(self):

################################################################################

# test code of the above class
if __name__ == '__main__':
    #test = prop_dielec(Mfrom='res', Mfile='total_dipole_moments.res', dt=1.)
    test = prop_dielec(Mfrom='traj', Mfile='1conf_H2O_sample.trj', dt=1.)
