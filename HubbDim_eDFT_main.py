############################################################
### A Python script with all subroutines for calculating ###
### basic quantities in the Hubbard dimer, such as       ###
### individual energies and densities, as well as exact  ###
### and approximate ensemble functionals                 ###
### Created by: Filip Cernatic                           ###
### e-mail: filip.cernatic@gmail.com                     ###
############################################################

# Python imports
from math import *
import numpy as np
from scipy.optimize import fminbound

# Individual energies
def Energy(twot,bigu,deltav,level="0_N"):
    """
    Calculates individual energies of all (Singlet) subspaces in
    the Hubbard dimer.

    :param twot: two times the hopping parameter
    :param bigu: electron-electron repulsion parameter on the same site
    :param deltav: site-potential difference in the Hubbard dimer
    :param level: energy level to-be-calculated
    :return: 
    """

    # Defining some intermediate parameters (r,theta)
    # which we need later for computing the individual energies.
    r = sqrt(3*((twot)**2+deltav**2)+bigu**2)
    theta = (1.0 / 3.0) * acos((9.0 * bigu * ((deltav ** 2) - 2 * ((twot / 2) ** 2)) - bigu ** 3) / (r ** 3))

    # Individual energies
    # 2-electron sector
    if level=="0_N":
        result = (2.0 / 3.0) * (bigu) + (2.0 / 3.0)*r*cos(theta + ((2 * pi) / 3))
    elif level=="1_N":
        result = (2.0 / 3.0) * (bigu) + (2.0 / 3.0)*r*cos(theta + ((2 * pi) / 3)*2)
    elif level=="2_N":
        result = (2.0 / 3.0) * (bigu) + (2.0 / 3.0)*r*cos(theta + ((2 * pi) / 3)*3)
    # 1- and 3-electron sectors (ground-state energies only)
    elif level=="0_Nm1":
        result = -(1/2)*sqrt((twot**2) + deltav**2)
    elif level=="0_Np1":
        result = -(1 / 2) * sqrt((twot ** 2) + deltav ** 2) + bigu
    else:
        print("Unknown level specified!")

    return result