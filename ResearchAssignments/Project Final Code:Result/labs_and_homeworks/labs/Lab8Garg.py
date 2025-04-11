import numpy as np
from astropy import units as u
from astropy import constants as const

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
%matplotlib inline
def StarFormationRate(L, Type, TIR=0):
    ''' Function that computes the star formation rate of a galaxy following Kennicutt & Evans 2012 Eq 12 (ARA&A 50)

    PARAMETERS
    -----------
    L: `float`
        Luminosity of the galaxy in erg/s
    Type: `string`
        The wavelength: `FUV`, `NUV`, `TIR`, `Halpha`
    TIR: `float`
        Total Infrared Luminosity in erg/s (default = 0)

    OUTPUTS
    --------
    SFR: `float
        Log of the Star Formation Rate (Msun/yr)
    '''

    if (Type == 'FUV'):
        logCx = 43.35 # Calibration from LFUV to SFR
        TIRc = 0.46 # Correction for dust absorption
    elif (Type == 'NUV'):
        logCx = 43.17 
        TIRc = 0.27
    elif (Type == 'Halpha'):
        logCx = 41.27
        TIRc = 0.0024
    elif (Type == 'TIR'):
        logCx = 43.41
        TIRc = 0
    else:
        print('WARNING: Missing the Wavelength. \
              I was expecting "FUV", "NUV", "Haplha", and "TIR"')
        
    # Correct the luminosity for dust using the TIR
    Lcorr = L + TIRc * TIR

    # Star formation rate
    SFR = np.log10(Lcorr) - logCx

    return SFR
# First need the Luminosity of the Sun in the right units
const.L_sun
#  WLM Dwarf Irregular Galaxy
#  WLM Dwarf Irregular Galaxy
LsunErgS = const.L_sun.to(u.erg/u.s).value
print(LsunErgS)
# WLM Dwarf Irregular Galaxy
# From NED GALEX DATA

NUV_WLM = 1.71e7 * LsunErgS
TIR_WLM = 2.48e6 * LsunErgS + 3.21e5 * LsunErgS + 2.49e6 * LsunErgS
# TIR = NIR + MIR + FIR
StarFormationRate(NUV_WLM, 'NUV', TIR_WLM)
def SFRMainSequence(Mstar, z):
    '''Function that computes the average SFR of a galaxy as a function of stellar mass and redshift
    
    PARAMETERS
    -----------
    Mstar: `float`
        Stellar mass of the galaxy in Msun
    z: `float`
        Redshift
        
    OUTPUTS
    --------
    SFR: `float`
        Log of the SFR (Msun/yr)
    '''

    alpha = 0.7 - 0.13 * z
    beta = 0.38 + 1.14 * z - 0.19 * z ** 2

    SFR = alpha * (np.log10(Mstar) - 10.5) + beta

    return SFR
# MW stellar mass (disk) at z=0
MWmstar = 7.5e10
# SFR for a MW type galaxy
print(SFRMainSequence(MWmstar, 0))
print(10**SFRMainSequence(MWmstar, 0))
# MW at z = 1
print(SFRMainSequence(MWmstar, 1))
print(10**SFRMainSequence(MWmstar, 1))
# create an array of stellar masses
Mass = np.linspace(1e8, 1e12)
fig = plt.figure(figsize=(8,8), dpi=500)
ax = plt.subplot(111)

# add log log plots
plt.plot(np.log10(Mass), SFRMainSequence(Mass, 0), color = 'blue', linewidth = 3, label = 'z = 0')
plt.plot(np.log10(Mass), SFRMainSequence(Mass, 1), color = 'red', linestyle = ':', linewidth = 3, label = 'z = 1')
plt.plot(np.log10(Mass), SFRMainSequence(Mass, 2), color = 'green', linestyle = '--', linewidth = 3, label = 'z = 2')
plt.plot(np.log10(Mass), SFRMainSequence(Mass, 3), color = 'purple', linestyle = '-.', linewidth = 3, label = 'z = 3')

# Add axis labels
plt.xlabel('Log(Mstar (M$_\odot$))', fontsize=12)
plt.ylabel('Log(SFR (M$_\odot$/year))', fontsize=12)


#adjust tick label font size
label_size = 12
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

# add a legend with some customizations.
legend = ax.legend(loc='upper left',fontsize='x-large')


# Save file
plt.savefig('Lab8_SFR_MainSequence.png')
# normal galaxies 
TIR_Normal = 1e10 * LsunErgS
print(10**StarFormationRate(TIR_Normal, "TIR"))
# LIRGs  
TIR_HLIRG = 1e11 * LsunErgS
print(10**StarFormationRate(TIR_HLIRG, "TIR"))
# ULIRGs
TIR_ULIRGS = 1e12 * LsunErgS
print(10**StarFormationRate(TIR_ULIRGS, "TIR"))
# HLIRGs
TIR_HLIRG = 1e13 * LsunErgS
print(10**StarFormationRate(TIR_HLIRG, "TIR"))
import os

html_file = "Lab8 Garg.html"
ipynb_file = "Lab8 Garg.ipynb"

# Delete old HTML if it exists
if os.path.exists(html_file):
    os.remove(html_file)

# Convert notebook to HTML
!jupyter nbconvert --to html "{ipynb_file}"

