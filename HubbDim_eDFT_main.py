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
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fminbound

# Individual energies
def energy(twot,bigu,deltav,level="0_N"):
    """
    Calculates individual energies of all (Singlet) subspaces in
    the Hubbard dimer.

    Parameters
    ----------
    twot: two times the hopping parameter
    bigu: electron-electron repulsion parameter on the same site
    deltav: site-potential difference in the Hubbard dimer
    level: energy level to-be-calculated

    Returns
    -------
    Energy of the specified level
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
    elif level=="0_N-1":
        result = -(1/2)*sqrt((twot**2) + deltav**2)
    elif level=="0_N+1":
        result = -(1 / 2) * sqrt((twot ** 2) + deltav ** 2) + bigu
    else:
        print("Unknown level specified!")

    return result

# Individual densities
def density(twot,bigu,deltav,level="0_N"):
    """
    Calculates individual densities of all (Singlet) subspaces in
    the Hubbard dimer.

    Parameters
    ----------
    twot: two times the hopping parameter
    bigu: electron-electron repulsion parameter on the same site
    deltav: site-potential difference in the Hubbard dimer
    level: energy level of the density to-be-calculated

    Returns
    -------
    Density (site occupation of site 0 of the specified level)
    """

    # Calculating some intermediate parameter
    t = twot/2

    # Then, we calculate individual densities
    if level=="0_N":
        E0_N = energy(twot, bigu, deltav, level="0_N")
        result = n0_N = 1 - (2 * deltav * E0_N) / (3 * (E0_N ** 2) - 4 * bigu * E0_N + bigu ** 2 - 4 * (t ** 2) - deltav ** 2)
    elif level=="1_N":
        E1_N = energy(twot, bigu, deltav, level="1_N")
        result = 1 - (2 * deltav * E1_N) / (3 * (E1_N ** 2) - 4 * bigu * E1_N + bigu ** 2 - 4 * (t ** 2) - deltav ** 2)
    elif level=="2_N":
        E2_N = energy(twot, bigu, deltav, level="2_N")
        result = 1 - (2 * deltav * E2_N) / (3 * (E2_N ** 2) - 4 * bigu * E2_N + bigu ** 2 - 4 * (t ** 2) - deltav ** 2)
    elif level=="0_N-1":
        result = (1 / 2) * (1 + (deltav / np.sqrt(4 * (t ** 2) + deltav ** 2)))
    elif level=="0_N+1":
        result = (1 / 2) * (3 + (deltav / np.sqrt(4 * (t ** 2) + deltav ** 2)))

    return result

# Ensemble energy
def ensemble_energy(twot,bigu,deltav,xim,xi):
    """
    Calculates the Ensemble energy of the ensemble of N,N-1 electron ground states
    and the first excited state in the Hubbard dimer.

    Parameters
    ----------
    twot: two times the hopping parameter
    bigu: electron-electron repulsion parameter on the same site
    deltav: site-potential difference in the Hubbard dimer
    xim: xi_minus - ensemble weight of the N-1 - electron ground state in the ensemble
    xi: ensemble weight of the N - electron excited state in the ensemble

    Returns
    -------
    Ensemble energy as a `float`.
    """

    E0_N = energy(twot,bigu,deltav,level="0_N")
    E1_N = energy(twot,bigu,deltav,level="1_N")
    E0_Nm1 = energy(twot,bigu,deltav,level="0_N-1")

    result = (1-(xim/2)-xi)*E0_N + xim*E0_Nm1 + xi*E1_N

    return result

# Ensemble density
def ensemble_density(twot,bigu,deltav,xim,xi):
    """
    Calculates the Ensemble density of the ensemble of N,N-1 electron ground states
    and the first excited state in the Hubbard dimer.

    Parameters
    ----------
    twot: two times the hopping parameter
    bigu: electron-electron repulsion parameter on the same site
    deltav: site-potential difference in the Hubbard dimer
    xim: xi_minus - ensemble weight of the N-1 - electron ground state in the ensemble
    xi: ensemble weight of the N - electron excited state in the ensemble

    Returns
    -------
    Ensemble density as a `float`.
    """

    n0_N = density(twot,bigu,deltav,level="0_N")
    n1_N = density(twot,bigu,deltav,level="1_N")
    n0_Nm1 = density(twot,bigu,deltav,level="0_N-1")

    result = (1-(xim/2)-xi)*n0_N + xim*n0_Nm1 + xi*n1_N

    return result

###########################
# EXACT FUNCTIONALS
###########################

# The exact ensemble Hartree-exchange-correlation functional, determined as a Legendre-Fenchel transform
def ensemble_Hxc(twot, bigu, xim, xi, n):
    """
    The numerically optimized version of the exact ensemble
    Hartree-exchange-correlation functional.

    Parameters
    ----------
    twot: two times the hopping parameter
    bigu: electron-electron repulsion parameter on the same site
    xim: xi_minus - ensemble weight of the N-1 - electron ground state in the ensemble
    xi: ensemble weight of the N - electron excited state in the ensemble
    n: density on site 0, can be an arbitrary value between 0 and 2, but the
    functional is exact at the exact ensemble density as an input!

    Returns
    -------
    Exact ensemble Hxc functional value at a given density, as a `float`.
    """

    # First, we numerically optimize the universal functional
    deltav_max = fminbound(lambda x: -(ensemble_energy(twot,bigu,x,xim,xi) + x * (n - 1)), -1000, 1000)

    F_ens = ensemble_energy(twot, bigu, deltav_max, xim, xi) + deltav_max * (n - 1)

    # Then, we optimize the ensemble KS functional
    deltavKS_max = fminbound(lambda x: -(ensemble_energy(twot, 0, x, xim, xi) + x * (n - 1)), -1000, 1000)

    Ts_ens = ensemble_energy(twot, 0, deltavKS_max, xim, xi) + deltavKS_max * (n - 1)

    # And finally, we compute the desired result, i.e. the Hartree-Exchange-correlation functional
    result = F_ens - Ts_ens

    return result

# The ensemble noninteracting kinetic energy functional
def ensemble_Ts(twot,xi,n):
    """
    The ensemble noninteracting kinetic energy functional.

    Parameters
    ----------
    twot: two times the hopping parameter
    xi: ensemble weight of the N - electron excited state in the ensemble
    n: density on site 0, can be an arbitrary value between 0 and 2, but the
    functional is exact at the exact ensemble density as an input!

    Returns
    -------
    Ensemble noninteracting kinetic energy value at a given density, as a `float`.
    """

    result = -(twot)*np.sqrt((1-xi)**2 - (1-n)**2)

    return result

# The exact ensemble Hartree-exchange functional
def ensemble_Hx(bigu,xim,xi,n):
    """
    The exact ensemblee Hartree-exchange functional.

    Parameters
    ----------
    bigu: electron-electron repulsion parameter on the same site
    xim: xi_minus - ensemble weight of the N-1 - electron ground state in the ensemble
    xi: ensemble weight of the N - electron excited state in the ensemble
    n: density on site 0, can be an arbitrary value between 0 and 2, but the
    functional is exact at the exact ensemble density as an input!

    Returns
    -------
    Exact ensemble Hx functional value at a given density, as a `float`.
    """

    result = (bigu / 2) * (1 + xi - (xim / 2) + (1 - 3 * xi - (xim / 2)) * (((1 - n) / (1 - xi)) ** 2))

    return result


###########################
# APPROXIMATE FUNCTIONALS
###########################

# The U-PT2 correlation energy
def ensemble_Corr_UPT2(twot,bigu,xim,xi,n):
    """
    The 'exact' ensemble U-PT2 functional, i.e. the approximation
    to the correlation energy functional as a second-order term of
    the perturbation expansion of the Universal HK functional as a
    Legendre-Fenchel transform around U=0.

    Parameters
    ----------
    twot: two times the Hopping parameter
    bigu: electron-electron repulsion parameter on the same site
    xim: xi_minus - ensemble weight of the N-1 - electron ground state in the ensemble
    xi: ensemble weight of the N - electron excited state in the ensemble
    n: density on site 0, can be an arbitrary value between 0 and 2, but the
    functional is exact at the exact ensemble density as an input!

    Returns
    -------
    Ensemble "U-PT2" correlation functional at a given density, as a `float`.
    """

    # Computing intermediate terms first
    term1 = -(((bigu)**2)/(8*twot))*(1-xi-(xim/2))*( (((1-xi)**2 - (n-1)**2)**(3/2))/((1-xi)**3) )
    term2 = (1 + (((n-1)/(1-xi))**2)*(3 - 4*( ((1-3*xi-(xim/2))**2)/((1-xi-(xim/2))*(1-xi))  )))

    result = term1*term2

    return result

# The strongly-correlated dimer correlation energy functional
def ensemble_Corr_StrongCorr(bigu,xim,xi,n):
    """
    This correlation energy functional is only applicable
    to weakly asymmetric and strongly correlated Hubbard dimers,
    where ∆v<<t<<U. Otherwise, the correlation energy is horribly wrong.

    Parameters
    ----------
    bigu: electron-electron repulsion parameter on the same site
    xim: xi_minus - ensemble weight of the N-1 - electron ground state in the ensemble
    xi: ensemble weight of the N - electron excited state in the ensemble
    n: density on site 0, can be an arbitrary value between 0 and 2, but the
    functional is exact at the exact ensemble density as an input!

    Returns
    -------
    Ensemble correlation energy approximation `float`.
    """

    result = bigu*max(xi,abs(n-1))-ensemble_Hx(bigu,xim,xi,n)

    return result

#####################
# PHYSICAL PROPERTIES
#####################

def optical_gap_LIM(twot,bigu,deltav,method="exact"):


    if method == "exact":
        result = 2*(ensemble_energy(twot,bigu,deltav,0,1/2)-ensemble_energy(twot,bigu,deltav,0,0))
    else:
        if method == "approximate_EEXX":
            a,b,c,d = (1,1,0,0)
        elif method == "approximate_Ec_UPT2":
            a,b,c,d = (1,1,1,0)
        elif method == "approximate_Ec_strongcorr":
            a,b,c,d = (1,1,0,1)

        n_xionehalf = ensemble_density(twot,bigu,deltav,0,1/2)
        ensemble_energy_approx_xionehalf = a*ensemble_Ts(twot,1/2,n_xionehalf) + b*ensemble_Hx(bigu,0,1/2,n_xionehalf)+\
            c*ensemble_Corr_UPT2(twot,bigu,0,1/2,n_xionehalf)+deltav*(1-n_xionehalf)+\
            d*ensemble_Corr_StrongCorr(bigu,0,1/2,n_xionehalf)
        n_xizero = ensemble_density(twot,bigu,deltav,0,0)
        ensemble_energy_approx_xizero = a*ensemble_Ts(twot,0,n_xizero) + b*ensemble_Hx(bigu,0,0,n_xizero)+\
            c*ensemble_Corr_UPT2(twot,bigu,0,0,n_xizero)+deltav*(1-n_xizero)+\
            d*ensemble_Corr_StrongCorr(bigu, 0, 0, n_xizero)

        result = 2*(ensemble_energy_approx_xionehalf-ensemble_energy_approx_xizero)

    return result




##################################################
# Example of a computation: Ensemble energy and
# LIM neutral excitation energy at various
# levels of approximation
##################################################

def ExampleCode():
    """
    Example of computation of ensemble energy in the asymmetric Hubbard dimer.
    Approximation levels included: Exact, EEXX and ECorr(UPT2)
    Returns
    -------
    The code returns two plots.
    Plot 1: Ensemble energy at various weights and various
    levels of approximation of the strongly correlated Hubbard dimer.
    Plot 2: Optical gap, extracted with the LIM method,
    at various correlation regimes of the asymmetric Hubbard dimer.
    """
    twot = 2
    bigu_plot1 = 5
    deltav = 0
    weightlist = np.linspace(0,1/2,100)
    bigulist = np.linspace(0,10,100)

    # Values to-be-plotted in plot 1
    optgap_exact_list = [optical_gap_LIM(twot,bigu,deltav,method="exact") for bigu in bigulist]
    optgap_EEXX_list = [optical_gap_LIM(twot,bigu,deltav,method="approximate_EEXX") for bigu in bigulist]
    optgap_EcUPT2_list = [optical_gap_LIM(twot, bigu, deltav, method="approximate_Ec_UPT2") for bigu in bigulist]
    #optgap_EcSc_list = [optical_gap_LIM(twot, bigu, deltav, method="approximate_Ec_strongcorr") for bigu in bigulist]

    # Plotting all the stuff
    fig,ax = plt.subplots(1,1,figsize=(8,5))

    ax.set_title("Optical gaps")
    ax.plot(bigulist,optgap_exact_list,color="red",label="exact")
    ax.plot(bigulist,optgap_EEXX_list,color="blue",ls=":",label="EEXX")
    ax.plot(bigulist,optgap_EcUPT2_list,color="lightblue",ls="-.",label="Ec_UPT2")
    #ax.plot(bigulist, optgap_EcSc_list, color="green", ls="--", label="Ec_strongcorr")

    ax.legend()

    plt.show()


# Running the example code if you call this library directly:
if __name__=="__main__":
    ExampleCode()