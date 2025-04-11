# ===== File: Lab1Garg.py =====
# Import Modules 
import numpy as np # import numpy
import astropy.units as u # import astropy units
from astropy import constants as const # import astropy constants
# 4.76 * mu * Ro = VLSR + vsun
def VLSR(Ro, mu = 6.379, vsun = 12.24 * u.km/u.s): # astropy way of signing unit
    ''' 
    This function will compute the velocity at the local standard of rest
        VLSR = 4.74 * mu * Ro - vsun
    Inputs: Ro (astropy units kpc) Distance from the Sun to the Galactic Center
            mu is the proper motion of Sag A* (mas/yr)
                The default is from Reid & Brunthaler 2004
            vsun (astropy units km/s) the peculiar motion of the Sun in v direction (Schonrich + 2010)
    Outputs: VLSR (astropy units km/s) The local standard of rest 
    '''
    VLSR = 4.74 * mu * (Ro/u.kpc) * u.km/u.s - vsun
    return VLSR
# Compute VLSR using Reid 2014
VLSR_Reid = VLSR(RoReid)
print(VLSR_Reid)
print(np.round(VLSR_Reid))
# Computer VLSR using Sparke & Gallagher 
VLSR_Sparke = VLSR(RoSparke)
print(VLSR_Sparke)
print(np.round(VLSR_Sparke)) 
# Compute VLSR using Gravity Collab (Abuter 2019)
VLSR_Abuter = VLSR(RoAbuter)
print(VLSR_Abuter)
print(np.round(VLSR_Abuter))
# Different values of the distance to the Galactic Center
RoReid = 8.34 * u.kpc # Reid + 2014
RoAbuter = 8.178 * u.kpc # GRAVITY Abuter + 2019
RoSparke = 7.8 * u.kpc # Sparke & Gallagher Text
# Orbital Period of the Sun
T_Abuter = TorbSun(RoAbuter, Vsun)
print(T_Abuter)
# orbital period = 2piR/V
def TorbSun(Ro, Vc): 
    ''' 
    Function that computes the orbital period of the Sun
        T = 2 pi R / V
    Inputs: 
        Ro (astropy quantity) distance to the Galactic Center from the Sun (kpc)
        Vc (astropy quantity) velocity of the sun in the "v" direction (in the direction of circular speed) (km/s)
    Outputs:
        T (astropy quantity) Orbital Period (Gyr)
    '''
    VkpcGyr = Vc.to(u.kpc/u.Gyr) # converting V to kpc/Gyr
    T = 2 * np.pi * Ro/VkpcGyr # orbital period

    return T
VsunPec = 12.24 * u.km/u.s # peculiar motion
Vsun = VLSR_Abuter + VsunPec # the total motion of the sun in "v" direction
AgeUniverse = 13.8 * u.Gyr
print (AgeUniverse/T_Abuter)
print(const.G)
Grav = const.G.to(u.kpc ** 3 / u.Gyr ** 2 / u.Msun)
print(Grav)
# Density profile rho = VLSR^2 / (4pi * G * R^2)
# Mass(r) = Integrate rho dV
#           Integrate rho 4pi * r^2 * dr
#           Integrate VLSR^2 / (4pi * G * r^2) * 4pi * r^2 dr
#           Integrate VLSR^2 / G dr
#           VLSR^2 / G * r

def massIso(r, VLSR):
    '''
    This function will compute the dark matter mass enclosed within a given distance, r, assuming an Isothermal Sphere Model
        M(r) = VLSR^2 / G * r
    Inputs: 
        r (astropy quanitity) distance from the Galactic Center (kpc)
        VLSR (astropy quantity) the velocity at the Local Standard of Rest (km/s)
    Outputs: 
        M (astropy quantity) mass enclosed within r (Msun)
    '''
    VLSRkpcGyr = VLSR.to(u.kpc/u.Gyr) # translating to kpc/Gyr
    M = VLSRkpcGyr ** 2 / Grav * r # Isothermal Sphere Mass Profile

    return M
# Compute the mass enclosed within 260 kpc
# Always assume constant velocity for this profile
mIso260 = massIso(260 * u.kpc, VLSR_Abuter)
print(f"{mIso260:.2e}")
# Compute the mass enclosed within Ro (Gravity Collab)
mIsoSolar = massIso(RoAbuter, VLSR_Abuter)
print(mIsoSolar)
print(f"{mIsoSolar:.2e}")
# Compute the mass enclosed within 260 kpc
mIsoIAU = massIso(260 * u.kpc, 220 * u.km / u.s)
print(f"{mIsoIAU:.2e}")
# Potential for a Hernquist Sphere
# Phi = -G * M / (r + a)
# Escape Speed becomes: vesc^2 = 2G * M / (r + a)
# rearrange for M: M = vesc^2/2/G(r + a)

def massHernVesc(vesc, r, a = 30 * u.kpc): # calling constant value have to be at the last position
    ''' 
    This function determines the total dark matter mass needed given an escape speed, assuiming a Hernquist profile
        M = vesc^2/2/G(r + a)

    Inputs: 
        vesc (astropy quantity) escaped speed (or speed of satellite) (km/s)
        r: (astropy quantity) distance from the Galactic Center (kpc)
        a: (astropy quantity) the Hernquist scale length (kpc) defaut value of 30 kpc
    Outputs:
        M (astropy quantity) mass within r (Msun)
    '''

    vescKpcGyr = vesc.to(u.kpc/u.Gyr) # translate to kpc/Gyr

    M = vescKpcGyr ** 2 /2/ Grav * (r + a)
    
    return M
Vleo = 196 * u.km/u.s # Speed of Leo I Sohn et al.
r = 260 * u.kpc
MLeoI = massHernVesc(Vleo, r)
print(f"{MLeoI:.2e}")
import os

html_file = "Lab1 Garg.html"
ipynb_file = "Lab1 Garg.ipynb"

# Delete old HTML if it exists
if os.path.exists(html_file):
    os.remove(html_file)

# Convert notebook to HTML
!jupyter nbconvert --to html "{ipynb_file}"



# ===== File: Lab2Garg.py =====
# ASTR 400 B 
# In Class Lab 2

# Import Modules 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad # For integration
# Documentation and examples for quad : 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
# https://www.tutorialspoint.com/scipy/scipy_integrate.htm

def schechter_M(m,phi_star=0.0166,m_star=-23.19,alpha=-0.81):
    """ Function that computes the Schechter Luminosity Function 
    for a given magnitude, assuming default parameters for field 
    galaxies in SDSS at z~0.1 in the Kband (Smith+2009)
    
    Inputs
        m : an array of floats
            an array of Kband magnitudes  (assumes -5*log(h) implicitly)
        phi_star:  float
            normalization of Schechter fxn (h^3 Mpc^-3)
        m_star:  float 
            knee of the Schechter fxn (K-band magnitude, 
            assumes -5*log(h) implicitly)
        alpha:  float
            faint end slope of the Schechter fxn
    
    Output:
        schechterM: float
            number density of galaxies (comoving units) 
            at the given magnitude m - 5*log(h)
            

    """
    # You should divide up long functions instead of writing them as one long set
    
    # Grouping all constants together
    a = 0.4*np.log(10)*phi_star
    
    # The Power Law, controlling the faint end slope
    b = 10**(0.4*(m_star-m)*(alpha+1.0)) 
    
    # The Exponential controlling the high mass end behavior
    c = np.exp(-10**(0.4*(m_star-m))) 
    
    # schechter function for the given magnitude
    schechterM = a*b*c 
    # i.e. don't do the below
    # return 0.4*np.log(10)*phistar*10**(0.4*(Mstar - M)*(alpha +1.0))*np.exp(-10**(0.4*(Mstar - M)))

    return schechterM
# Create an array to store Kband Magnitudes from -26 to -17
mk = np.arange(-26, -16.99, 0.1)
print(mk)
# Plot the Schechter Function

fig = plt.figure(figsize=(10,10))  # sets the scale of the figure
ax = plt.subplot(111) 

# Plot the default values (y axis log)
# ADD HERE
ax.semilogy(mk, schechter_M(mk), color = 'blue', linewidth = 5, label = 'Smith+09')

# Q2 solutions: change alpha
# ADD HERE
ax.semilogy(mk, schechter_M(mk, alpha = - 1.35), color = 'red', linewidth = 5, linestyle = ':', label = r'high $\alpha$')

ax.semilogy(mk, schechter_M(mk, alpha = - 0.6), color = 'green', linewidth = 5, linestyle = '--', label = r'low $\alpha$')


# Add labels
plt.xlabel(r'M$_k$ + 5Log($h$)', fontsize=22)
plt.ylabel(r'$\Phi$ (Mpc$^{-3}h^3$/mag)', fontsize=22)

#set axis limits
plt.xlim(-17,-26)

#adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

# add a legend with some customizations.
legend = ax.legend(loc='lower left',fontsize='x-large')

# Save to a file
plt.savefig('Schechter_M.png')
def schechter_L(lum, n_star=8e-3, l_star=1.4e10, alpha=-0.7):
    """ Function that computes the Schechter Luminosity Function
        for a given luminosity. Defaults are from Sparke & Gallagher 
    
    Inputs:
        lum: array of floats
            Array of Luminosities (Lsun)
        
        n_star: float
            Normalization of the Schechter fxn (h^3 Mpc^-3)
            
        l_star: float
            Characteristic luminosity (knee of the Schechter fxn) 
            in units of Lsun
            
        alpha: float
            Faint end slope
            
     Outputs:
         schechterL: float
             number density of galaxies for a given luminosity 
             (h^3 * Mpc^-3/Lsun)
    """
    # Break down the equation into parts
    a = n_star/l_star # constants
    b = np.exp(-lum/l_star) # bright end
    c = (lum/l_star) ** alpha # faint end
    
    schechterL = a * b * c
    
    return schechterL
# Understanding lambda functions
# Short cut -- defines and evaluates a function in one line ! 

# lambda says that a function follows, where the variables are a and b, 
# and the function to be evaluated is a*b
x = lambda a, b : a * b
print(x(5, 6))
# Example Usage of quad and lambda

# Version 1
print(quad(np.sin, 0, np.pi))

# Version 2
f = lambda x: np.sin(x)
print(quad(f, 0, np.pi))

# Version 3
def ex(x):
    return np.sin(x) 

print(quad(lambda x: ex(x), 0, np.pi))

# Useful for numerical integration
# First number is the integrated value, second is the error
# What fraction of the integrated luminosity density  lies above l*
# alpha = -0.7

# luminosity density above L*
l_upper = quad(lambda L: L * schechter_L(L), 1.4e10, 1e14)
print(l_upper[0])

# total luminosity density
l_total = quad(lambda L: L * schechter_L(L), 0.1, 1e14)
print(l_total[0])

# fraction of luminosity density above L*
ratio = l_upper[0] / l_total[0]
print("Ratio (>L*)/Ltotal", np.round(ratio, 3))
def imf(m, m_min=0.1, m_max=120, alpha=2.35):
    
    ''' Function that defines the IMF (default is Salpeter). 
        The function is normalized such that 
        it returns the fraction of stars within some mass 
        interval m_min to m_max.

        Inputs:
            m: array of floats 
                Array of stellar masses (Msun)
            m_min:  float
                minimum mass (Msun)
            m_max : float
                maximal mass (Msun)
            alpha : float
                power law. default is the Salpeter IMF
                
        Output:
            norm_imf: float
                normalized fraction of stars at a given m
    '''
    # Determine the normalization for the imf
    to_normalize = quad(lambda m: m ** (-alpha), m_min, m_max)

    # Normalization factor
    norm = 1/to_normalize[0]

    # Define the normalized imf
    norm_imf = norm * m ** (-alpha)
    
    return norm_imf
    
test = quad(lambda m: imf(m), 0.1, 120)
print(np.round(test[0], 3))
frac = quad(lambda m: imf(m), 1, 120)
print(np.round(frac[0], 3))
# cluster with 5000 stars 
5000 * np.round(frac[0], 3) # not accurate in terms of number of stars, only .1 solar mass
                            # Better said in term of mass
def imf_Mass(m, m_min=0.1, m_max=120, alpha=2.35):
    
    ''' Function that defines the IMF (default is Salpeter). 
        The function is normalized such that 
        it returns the fraction of mass within some range of mass 
        interval m_min to m_max.
        
        Inputs:
            m: array of floats 
                Array of stellar masses (Msun)
            m_min:  float
                minimum mass (Msun)
            m_max : float
                maximal mass (Msun)
            alpha : float
                power law. default is the Salpeter IMF
                
        Output:
            norm_imf_mass: float
                normalized fraction of mass over a given m range 
    '''
    # Determine the normalization for the imf
    to_normalize = quad(lambda m: m * m ** (-alpha), m_min, m_max)

    # Normalization factor
    norm = 1/to_normalize[0]

    # Define the normalized imf
    norm_imf_mass = norm * m * m ** (-alpha)
    
    return norm_imf_mass
    
# Determine the fraction of mass in stars that are more massive than the Sun
frac2 = quad(lambda m: imf_Mass(m), 1, 120)
print(np.round(frac2[0], 3))
# 5000 Msun cluster - how much mass is between 1 Msun and 120 Msun
print(5000 * np.round(frac2[0], 3))
import os

html_file = "Lab2 Garg.html"
ipynb_file = "Lab2 Garg.ipynb"

# Delete old HTML if it exists
if os.path.exists(html_file):
    os.remove(html_file)

# Convert notebook to HTML
!jupyter nbconvert --to html "{ipynb_file}"



# ===== File: Lab3Garg.py =====
#In Class Lab 3 Template
# G Besla ASTR 400B

# Load Modules
import numpy as np
import astropy.units as u

# import plotting modules
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

# Some Notes about the Isochrone Data
# DATA From   
# http://stellar.dartmouth.edu/models/isolf_new.html
# files have been modified from download.  
# ( M/Mo --> M;   Log L/Lo --> L)
# removed #'s from all lines except column heading
# NOTE SETTINGS USED:  
# Y = 0.245 default   [Fe/H] = -2.0  alpha/Fe = -0.2
# These could all be changed and it would generate 
# a different isochrone


# Filename for data with Isochrone fit for 1 Gyr
# These files are located in the folder IsochroneData
filename1="./IsochroneData/Isochrone1.txt"

# major peak
filename11 = './IsochroneData/Isochrone11.txt'
filename10 = './IsochroneData/Isochrone10.txt'

data11 = np.genfromtxt(filename11,dtype=None,
                      names=True,skip_header=8)

data10 = np.genfromtxt(filename10,dtype=None,
                      names=True,skip_header=8)
# next peak
filename6 = './IsochroneData/Isochrone6.txt'
filename7 = './IsochroneData/Isochrone7.txt'

data6 = np.genfromtxt(filename6,dtype=None,
                      names=True,skip_header=8)

data7 = np.genfromtxt(filename7, dtype=None,
                      names=True,skip_header=8)
# READ IN DATA
# "dtype=None" means line is split using white spaces
# "skip_header=8"  skipping the first 8 lines 
# the flag "names=True" creates arrays to store the date
#       with the column headers given in line 8 

# Read in data for an isochrone corresponding to 1 Gyr
data1 = np.genfromtxt(filename1,dtype=None,
                      names=True,skip_header=8)

# Plot Isochrones 
# For Carina

fig = plt.figure(figsize=(10,10))
ax = plt.subplot(111)

# Plot Isochrones

# Isochrone for 1 Gyr
# Plotting Color vs. Difference in Color 
plt.plot(data1['B']-data1['R'], data1['R'], color='blue', 
         linewidth=5, label='1 Gyr')
###EDIT Here, following the same format as the line above 
plt.plot(data10['B']-data10['R'], data10['R'], color='red', 
         linewidth=5, label='10 Gyr')
plt.plot(data11['B']-data11['R'], data11['R'], color='yellow', 
         linewidth=5, label='11 Gyr')
plt.plot(data6['B']-data6['R'], data6['R'], color='magenta', 
         linewidth=5, label='6 Gyr')
plt.plot(data7['B']-data7['R'], data7['R'], color='green', 
         linewidth=5, label='7 Gyr')

# Add axis labels
plt.xlabel('B-R', fontsize=22)
plt.ylabel('M$_R$', fontsize=22)

#set axis limits
plt.xlim(-0.5,2)
plt.ylim(5,-2.5)

#adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

# add a legend with some customizations.
legend = ax.legend(loc='upper left',fontsize='x-large')

#add figure text
plt.figtext(0.5, 0.15, 'CMD for Carina dSph', fontsize=22)

plt.savefig('IsochroneCarina.png')


import os

html_file = "Lab3 Garg.html"
ipynb_file = "Lab3 Garg.ipynb"

# Delete old HTML if it exists
if os.path.exists(html_file):
    os.remove(html_file)

# Convert notebook to HTML
!jupyter nbconvert --to html "{ipynb_file}"


# ===== File: Lab4Garg.py =====
# In Class Lab 4
# G. Besla 

# import relevant modules 
import astropy.units as u
import numpy as np
from astropy import constants as const # import astropy constants
def hernquist_mass(r,h_a=60*u.kpc, m_halo=1.975): # ADD m_halo=??
    """ Function that defines the Hernquist 1990 mass profile 
    Inputs:
        r: astropy quantity
            Galactocentric distance in kpc
        a: astropy quantity
            scale radius of the Hernquist profile in kpc
        m_halo: float
            total halo mass in units of 1e12 Msun 
        
    Ouputs:
        mass:  astropy quantity
            total mass within the input radius r in Msun
    """
    # constants, correcting units
    a = m_halo * 1e12 * u.Msun 
    b = r ** 2 / (h_a + r) ** 2
    
    mass = a * b  # Hernquist profile
    
    return mass
print(f"{hernquist_mass(1e5 * u.kpc):.2e}")
print(f"{hernquist_mass(260 * u.kpc):.2e}")
print(f"{hernquist_mass(50 * u.kpc):.2e}")
# Disk Mass
mdisk = 0.075e12 * u.Msun

# Bulge Mass
mbulge = 0.012e12 * u.Msun
# total mass of the MW within 50 kpc
mass_MW50 = mdisk + mbulge + hernquist_mass(50 * u.kpc)
print(f'{mass_MW50:.2e}')
# Rj = r * (Msat/2/Mmw) ** (1/3)
# (Rj/r) ** 3 = Msat/2/Mmw
# Msat = 2 * Mmw * (Rj/r) ** 3

def jacobi_mass(rj,r,m_host):
    """ Function that determines the minimum satellite
    mass needed to maintain a the size of a given 
    satellite using the Jacobi Radius
    
    Inputs:
        rj : astropy quantity
            Jacobi Radius or the stellar radius of the 
            satellite in kpc
        r : astropy quantity 
            Distance of the satellite from the host in kpc
        m_host: astropy quantity 
            Mass of the host galaxy in Msun within r in Msun
        
    Outputs:
        m_min: astropy quantity
            Minimum satellite mass in Msun
    """
    # constants 
    a = 2 * m_host 
    b = (rj/r) ** 3 

    m_min = a * b # min satellite mass
    
    return m_min
sizeL = 18.5 * u.kpc # observed size of the LMC
# Mackey + 2016

distL = 50 * u.kpc # Galactocentric distance to the LMC
# minimum mass of the LMC needed to maintain size of 18.5 kpc
LMC_jacobiM = jacobi_mass(sizeL, distL, mass_MW50)
print(f"{LMC_jacobiM:.2e}")

LMC_mstar = 3e9 * u.Msun
print(np.round(LMC_jacobiM / LMC_mstar))
import os

html_file = "Lab4 Garg.html"
ipynb_file = "Lab4 Garg.ipynb"

# Delete old HTML if it exists
if os.path.exists(html_file):
    os.remove(html_file)

# Convert notebook to HTML
!jupyter nbconvert --to html "{ipynb_file}"



# ===== File: Lab5Garg.py =====
# Import Modules 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy import constants as const # import astropy constants
import astropy.units as u
# Gravitational Constant in the desired units
# kpc^3/Gyr^2/Msun
Grav = const.G.to(u.kpc**3/u.Gyr**2/u.Msun)
def WolfMass(sigma, re):
    """ Function that defines the Wolf mass estimator from Wolf+ 2010
    PARAMETERS
    ----------
        sigma: astropy quantity
            1D line of sight velocity dispersion in km/s
        re: astropy quantity
            Effective radius, 2D radius enclosing half the
            stellar mass in kpc
    OUTPUTS
    -------
        mWolf: Returns the dynamical mass within the 
            half light radius in Msun
    """
    
    sigmaKpcGyr = sigma.to(u.kpc/u.Gyr) # velocity dispersion units
    
    mWolf = 4/Grav*sigmaKpcGyr**2*re # Wolf mass estimator
    
    return mWolf
# 47 Tuc Parameters
lumTuc = 1e5 * u.Lsun # luminosity
sigmaTuc = 17.3 * u.km/u.s # 1D los (line of sight) vel dispersion
reTuc = 0.5/1000 * u.kpc # effective radius (2D half  light)
# Dynamical mass for 47 Tuc
massTuc = WolfMass(sigmaTuc, reTuc)
print(f"{massTuc:.2e}")
# M/L of ~ 1 
print(f"Mass to Light Ratio of 47 Tuc: {np.around(massTuc/lumTuc, 1)}")
# William I Parameters
lumWI = 1e3 * u.Lsun # luminosity
sigmaWI = 4.3 * u.km/u.s # 1D los vel dispersion
reWI = 25/1000 * u.kpc # effective radius
# Dynamical Mass of Willman I
massWI = WolfMass(sigmaWI, reWI)
print(f"{massWI:.2e}")
# M/L of ~ 1 
print(f"Mass to Light Ratio of Willman I: {np.around(massWI/lumWI, 1)}")
class AbundanceMatching:
    """ Class to define the abundance matching relations from 
    Moster et al. 2013, which relate the stellar mass of a galaxy
    to the expected dark matter halo mass, according to 
    Lambda Cold Dark Matter (LCDM) theory """
    
    
    def __init__(self, mhalo, z):
        """ Initialize the class
        
        PARAMETERS
        ----------
            mhalo: float
                Halo mass in Msun
            z: float
                redshift
        """
        
        #initializing the parameters:
        self.mhalo = mhalo # Halo Mass in Msun
        self.z = z  # Redshift
        
        
    def logM1(self):
        """eq. 11 of Moster 2013
        OUTPUT: 
            M1: float 
                characteristic mass in log(Msun)
        """
        M10      = 11.59
        M11      = 1.195 
        return M10 + M11*(self.z/(1+self.z))  
    
    
    def N(self):
        """eq. 12 of Moster 2013
        OUTPUT: 
            Normalization for eq. 2
        """
        N10      = 0.0351
        N11      = -0.0247
    
        return N10 + N11*(self.z/(1+self.z))
    
    
    def Beta(self):
        """eq. 13 of Moster 2013
        OUTPUT:  power of the low mass slope"""
        beta10      = 1.376
        beta11      = -0.826
    
        return beta10 + beta11*(self.z/(1+self.z))
    
    def Gamma(self):
        """eq. 14 of Moster 2013
        OUTPUT: power of the high mass slope """
        gamma10      = 0.608
        gamma11      = 0.329
    
        return gamma10 + gamma11*(self.z/(1+self.z))
    
    
    def SHMratio(self):
        """ 
        eq. 2 of Moster + 2013
        The ratio of the stellar mass to the halo mass
        
        OUTPUT: 
            SHMratio float
                Stellar mass to halo mass ratio
        """
        M1 = 10**self.logM1() # Converting characteristic mass 
        # to Msun from Log(Msun)
        
        A = (self.mhalo/M1)**(-self.Beta())  # Low mass end
        
        B = (self.mhalo/M1)**(self.Gamma())   # High mass end
        
        Norm = 2*self.N() # Normalization
    
        SHMratio = Norm*(A+B)**(-1)
    
        return SHMratio 
    
 # Q1: add a function to the class that takes the SHM ratio and returns 
# The stellar mass 
    def StellarMass(self):
        '''
        Method to compute the stellar mass using eq. 2 of Moster + 2013 (stellar/halo mass ratio)

        OUTPUT:
            starMass: float, stellar mass in Msun
        '''
        starMass = self.mhalo * self.SHMratio()

        return starMass
mh = np.logspace(10,15,1000) # Logarithmically spaced array
# Define Instances of the Class for each redshift
MosterZ0 = AbundanceMatching(mh,0) # z = 0
MosterZ0_5 = AbundanceMatching(mh,0.5) # z = 0.5
MosterZ1 = AbundanceMatching(mh,1) # z = 1
MosterZ2 = AbundanceMatching(mh,2) # z = 2
fig,ax = plt.subplots(figsize=(10,8))


#adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

# Plot z = 0
plt.plot(np.log10(mh), np.log10(MosterZ0.StellarMass()),
         linewidth = 5, label='z=0')

# Continue plotting for the other redshifts here

# Plot z = 0.5
plt.plot(np.log10(mh), np.log10(MosterZ0_5.StellarMass()),
         linewidth = 5, linestyle = 'dotted', label='z=0.5')

# Plot z = 1
plt.plot(np.log10(mh), np.log10(MosterZ1.StellarMass()),
         linewidth = 5, linestyle = 'dashdot', label='z=1')

# Plot z = 2
plt.plot(np.log10(mh), np.log10(MosterZ2.StellarMass()),
         linewidth = 5, linestyle = 'dashed', label='z=2')


# Axes labels 
plt.xlabel('log (M$_h$/M$_\odot$)',fontsize=22) 
plt.ylabel('log (m$_\star$/M$_\odot$)', fontsize=22)

# Legend
plt.legend(loc='lower right',fontsize='x-large')

# save the file 
plt.savefig('AbundanceMatching_Lab5.png')
# LMC halo mass
haloLMC1 = 3e10 # traditional models

# Abundance matching object
LMC1 = AbundanceMatching(haloLMC1, 0)
# Find the stellar mass
LMC1star = LMC1.StellarMass()

print(LMC1star/1e9, 3)
print(LMC1star/3e9 * 100)
# say we know that LMC stellar mass = 3e9 Msun
#  what is the halo mass?

haloLMC2 = 17e10

LMC2 = AbundanceMatching(haloLMC2, 0)
LMC2star = LMC2.StellarMass()

print(np.round(LMC2star/1e9, 3))
# Find the characteristic Halo mass at z = 0

M1halo_z0 = MosterZ0.logM1()
print(f'Log M1, z = 0: {M1halo_z0}')
# Create a new instance of the class, with halo mass = log M1 at z = 0

M1z0 = AbundanceMatching(10 ** M1halo_z0, 0)
# Determine the stellar mass of that halo
M1star_z0 = M1z0.StellarMass()
print(f'Stellar mass of L * gal at z = 0: {M1star_z0/1e10} (1e10 M sun)')
# Repeating at z = 2

M1halo_z2 = MosterZ2.logM1()
print(f'Log M1, z = 2: {M1halo_z2}')
# Create a new instance of the class, with halo mass = log M1 at z = 2

M1z2 = AbundanceMatching(10 ** M1halo_z2, 2)
# Determine the stellar mass of that halo
M1star_z2 = M1z2.StellarMass()
print(f'Stellar mass of L * gal at z = 2: {M1star_z2/1e10} (1e10 M sun)')
import os

html_file = "Lab5 Garg.html"
ipynb_file = "Lab5 Garg.ipynb"

# Delete old HTML if it exists
if os.path.exists(html_file):
    os.remove(html_file)

# Convert notebook to HTML
!jupyter nbconvert --to html "{ipynb_file}"



# ===== File: Lab6Garg.py =====
# In Class Lab 6
# Surface Brightness Profiles

# Load Modules
import numpy as np
import astropy.units as u

# import plotting modules
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

# my modules
from ReadFile import Read
from CenterOfMass import CenterOfMass
from MassProfile import MassProfile
from GalaxyMass import ComponentMass
# didnt need the path comands all set because files already exist in the same folder for me
# Create a center of mass object for M31
# I.e. an instance of the CenterOfMass class 
M31_COM = CenterOfMass(r"M31_000.txt", 3)
# Center of Mass of M31
M31_COM_p = M31_COM.COM_P(0.1)
# Use the center of mass object to 
# store the x, y, z, positions and mass of the bulge particles
# be sure to correct for the COM position of M31
x = M31_COM.x - M31_COM_p[0].value
y = M31_COM.y - M31_COM_p[1].value
z = M31_COM.z - M31_COM_p[2].value
m = M31_COM.m # units of 1e10
# Determine the positions of the bulge particles in 
# cylindrical coordinates. 
cyl_r = np.sqrt(x**2 + y**2) # radial
cyl_theta = np.arctan2(y, x) # theta
def SurfaceDensity(r,m):
    """ Function that computes the surface mass density profile
    given an array of particle masses and radii 
     
    PARMETERS
    ---------
        r : array of `floats` - cyclindrical radius [kpc]
        m : array of `floats` - particle masses [1e10 Msun] 
    
    RETURNS
    -------
        r_annuli : array of `floats` -  radial bins for the 
            annuli that correspond to the surface mass density profile
    
        sigma: array of `floats` - surface mass density profile 
         [1e10 Msun/kpc^2] 
        
        
    """
    
    # Create an array of radii that captures the extent of the bulge
    # 95% of max range of bulge
    radii = np.arange(0.1, 0.95 * r.max(), 0.1)

    # create a mask to select particles within each radius
    # np.newaxis creates a virtual axis to make cyl_r_mag 2 dimensional
    # so that all radii can be compared simultaneously
    # a way of avoiding a loop - returns a boolean 
    enc_mask = r[:, np.newaxis] < radii

    # calculate mass of bulge particles within each annulus.  
    # relevant particles will be selected by enc_mask (i.e., *1)
    # outer particles will be ignored (i.e., *0)
    # axis =0 flattens to 1D
    m_enc = np.sum(m[:, np.newaxis] * enc_mask, axis=0)

    # use the difference between m_enc at adjacent radii 
    # to get mass in each annulus
    m_annuli = np.diff(m_enc) # one element less then m_enc
    
    
    # Surface mass density of stars in the annulus
    # mass in annulus / surface area of the annulus. 
    # This is in units of 1e10
    sigma = m_annuli / (np.pi * (radii[1:]**2 - radii[:-1]**2))
    # array starts at 0, but here starting at 1 and
    # subtracting radius that ends one index earlier.
    
    # Define the range of annuli
    # here we choose the geometric mean between adjacent radii
    r_annuli = np.sqrt(radii[1:] * radii[:-1]) 

    return r_annuli, sigma
# Define the surface mass density profile for the simulated bulge
# and the corresponding annuli
r_annuli, sigmaM31bulge = SurfaceDensity(cyl_r, m)

def sersicE(r, re, n, mtot):
    """ Function that computes the Sersic Profile for an Elliptical 
    System, assuming M/L ~ 1. As such, this function is also the 
    mass surface density profile. 
    
    PARMETERS
    ---------
        r: `float`
            Distance from the center of the galaxy (kpc)
        re: `float`
            The Effective radius (2D radius that contains 
            half the light) (kpc)
        n:  `float`
            sersic index
        mtot: `float`
            the total stellar mass (Msun)

    RETURNS
    -------
        I: `array of floats`
            the surface brightness/mass density (M/L = 1 for elliptical gal)
            profile for an elliptical in Lsun/kpc^2

    """
    # M/L = 1
    lum = mtot

    # the effective surface brightness
    Ie = lum/7.2/np.pi/re**2

    # break down the sersic profile
    a = (r/re) ** (1/n)
    b = -7.67 * (a - 1)

    # the surface brightness or mass density profile
    I = Ie * np.exp(b)
    
    return I
# Create a mass profile object for M31
# using solution to Homework 5
M31mass = MassProfile("M31", 0)
# Determine the Bulge mass profile
# use the annuli defined for the surface mass density profile
bulge_mass = M31mass.massEnclosed(3, r_annuli).value

# Determine the total mass of the bulge
bulge_total = ComponentMass("M31_000.txt", 3) * 1e12
print(f'{bulge_total:.2e}')
# Find the effective radius of the bulge, 
# Re encloses half of the total bulge mass

# Half the total bulge mass
b_half = bulge_total/2.0
print(f'{b_half:.2e}')
# Find the indices where the bulge mass is larger than b_half
index = np.where(bulge_mass > b_half)

# take first index where Bulge Mass > b_half
# check : should match b_half
print(f"{bulge_mass[index][0]:.2e}")
# Define the Effective radius of the bulge
re_bulge = r_annuli[index][0] * 3/4
print(re_bulge)
# Sersic Index = 4
SersicM31Bulge = sersicE(r_annuli, re_bulge, 4, bulge_total)
fig, ax = plt.subplots(figsize=(9, 8))

#adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size


# Surface Density Profile
# YOU ADD HERE
ax.loglog(r_annuli, sigmaM31bulge, lw = 2, label = 'Sim bulge')

# Sersic fit to the surface brightness Sersic fit
# YOU ADD HERE
ax.loglog(r_annuli, SersicM31Bulge/1e10, linestyle = '-.', lw = 3, label = 'Sersic n=4')


plt.xlabel('log r [ kpc]', fontsize=22)

# note the y axis units
plt.ylabel(r'log $\Sigma_{bulge}$ [$10^{10} M_\odot$ / kpc$^2$]', 
          fontsize=22)

plt.title('M31 Bulge', fontsize=22)

#set axis limits
plt.xlim(1,50)
plt.ylim(1e-5,0.1)

ax.legend(loc='best', fontsize=22)
fig.tight_layout()

plt.savefig('Lab6.png')
import os

html_file = "Lab6 Garg.html"
ipynb_file = "Lab6 Garg.ipynb"

# Delete old HTML if it exists
if os.path.exists(html_file):
    os.remove(html_file)

# Convert notebook to HTML
!jupyter nbconvert --to html "{ipynb_file}"



# ===== File: Lab7Garg.py =====
# In Class Lab 7 Template

# G. Besla
# with code from R. Hoffman, R. Li and E. Patel

# import modules
import numpy as np
import astropy.units as u
from astropy.constants import G

# import plotting modules
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# my modules
from ReadFile import Read
from CenterOfMass import CenterOfMass
from MassProfile import MassProfile

# for contours
import scipy.optimize as so
import sys

# Change path to homework3 where the ReadFile.py file is
module_path = "ReadFile.py"

# Add the directory to sys.path
sys.path.append(module_path)

# Import ReadFile
from ReadFile import Read
# Change path to homework4 where the class CenterOfMass file is
module_path = "CenterOfMass.py"

# Add the directory to sys.path
sys.path.append(module_path)

# Import CenterOfMass class
from CenterOfMass import CenterOfMass
# Code for plotting contours
# from https://gist.github.com/adrn/3993992


def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level

def density_contour(xdata, ydata, nbins_x, nbins_y, ax=None, **contour_kwargs):
    """ Create a density contour plot.
    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins_x : int
        Number of bins along x dimension
    nbins_y : int
        Number of bins along y dimension
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
        
    Example Usage
    -------------
     density_contour(x pos, y pos, contour res, contour res, axis, colors for contours)
     e.g.:
     density_contour(xD, yD, 80, 80, ax=ax, 
         colors=['red','orange', 'yellow', 'orange', 'yellow'])

    """

    H, xedges, yedges = np.histogram2d(xdata, ydata, bins=(nbins_x,nbins_y), density=True)
    # NOTE : if you are using the latest version of python, in the above: 
    # instead of normed=True, use density=True
    
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins_x))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y,1))

    pdf = (H*(x_bin_sizes*y_bin_sizes))
    
    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T
    fmt = {}
    
    ### Adjust Here #### 
    
    # Contour Levels Definitions
    one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68))
    two_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.95))
    three_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.99))
    
    # You might need to add a few levels


    # Array of Contour levels. Adjust according to the above
    levels = [one_sigma, two_sigma, three_sigma][::-1]
    
    # contour level labels  Adjust accoding to the above.
    strs = ['0.68','0.95', '0.99'][::-1]

    
    ###### 
    
    if ax == None:
        contour = plt.contour(X, Y, Z, levels=levels, origin="lower", **contour_kwargs)
        for l, s in zip(contour.levels, strs):
            fmt[l] = s
        plt.clabel(contour, contour.levels, inline=True, fmt=fmt, fontsize=12)

    else:
        contour = ax.contour(X, Y, Z, levels=levels, origin="lower", **contour_kwargs)
        for l, s in zip(contour.levels, strs):
            fmt[l] = s
        ax.clabel(contour, contour.levels, inline=True, fmt=fmt, fontsize=12)
    
    return contour
# Create a COM of object for M31 Disk (particle type=2) Using Code from Homework 4
COMD = CenterOfMass("M31_000.txt",2)
# Compute COM of M31 using disk particles
COMP = COMD.COM_P(0.1)
COMV = COMD.COM_V(COMP[0],COMP[1],COMP[2])
# Determine positions of disk particles relative to COM 
xD = COMD.x - COMP[0].value 
yD = COMD.y - COMP[1].value 
zD = COMD.z - COMP[2].value 

# total magnitude
rtot = np.sqrt(xD**2 + yD**2 + zD**2)

# Determine velocities of disk particles relatiev to COM motion
vxD = COMD.vx - COMV[0].value 
vyD = COMD.vy - COMV[1].value 
vzD = COMD.vz - COMV[2].value 

# total velocity 
vtot = np.sqrt(vxD**2 + vyD**2 + vzD**2)

# Arrays for r and v 
r = np.array([xD,yD,zD]).T # transposed 
v = np.array([vxD,vyD,vzD]).T
# 1) Make plots 

# M31 Disk Density 
fig, ax= plt.subplots(figsize=(12, 10))

# ADD HERE
# plot the particle density for M31 using a 2D historgram
# plt.hist2D(pos1,pos2, bins=, norm=LogNorm(), cmap='' )
# cmap options: 
# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html  
#   e.g. 'magma', 'viridis'
# can modify bin number to make the plot smoother
plt.hist2d(xD, yD, bins = 150, norm = LogNorm(), cmap = 'magma')

cbar = plt.colorbar()
cbar.set_label("Number of disk particle per bin", fontsize=15)

# ADD HERE
# make the contour plot
# x pos, y pos, contour res, contour res, axis, colors for contours.
# remember to adjust this if there are other contours added
# density_contour(pos1, pos2, res1, res2, ax=ax, colors=[])
density_contour(xD, yD, 80, 80, ax=ax, colors=['red', 'yellow', 'white', 'yellow'])


# Add axis labels
plt.xlabel('x (kpc) ', fontsize=22)
plt.ylabel('y (kpc) ', fontsize=22)

#set axis limits
plt.ylim(-40,40)
plt.xlim(-40,40)

#adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size



# Save to a file
plt.savefig('Lab7_M31Disk.png')
def RotateFrame(posI,velI):
    """a function that will rotate the position and velocity vectors
    so that the disk angular momentum is aligned with z axis. 
    
    PARAMETERS
    ----------
        posI : `array of floats`
             3D array of positions (x,y,z)
        velI : `array of floats`
             3D array of velocities (vx,vy,vz)
             
    RETURNS
    -------
        pos: `array of floats`
            rotated 3D array of positions (x,y,z) 
            such that disk is in the XY plane
        vel: `array of floats`
            rotated 3D array of velocities (vx,vy,vz) 
            such that disk angular momentum vector
            is in the +z direction 
    """
    
    # compute the angular momentum
    L = np.sum(np.cross(posI,velI), axis=0)
    
    # normalize the angular momentum vector
    L_norm = L/np.sqrt(np.sum(L**2))


    # Set up rotation matrix to map L_norm to
    # z unit vector (disk in xy-plane)
    
    # z unit vector
    z_norm = np.array([0, 0, 1])
    
    # cross product between L and z
    vv = np.cross(L_norm, z_norm)
    s = np.sqrt(np.sum(vv**2))
    
    # dot product between L and z 
    c = np.dot(L_norm, z_norm)
    
    # rotation matrix
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    v_x = np.array([[0, -vv[2], vv[1]], [vv[2], 0, -vv[0]], [-vv[1], vv[0], 0]])
    R = I + v_x + np.dot(v_x, v_x)*(1 - c)/s**2

    # Rotate coordinate system
    pos = np.dot(R, posI.T).T
    vel = np.dot(R, velI.T).T
    
    return pos, vel
# ADD HERE
# compute the rotated position and velocity vectors
rn, vn = RotateFrame(r, v)
# Rotated M31 Disk - EDGE ON

# M31 Disk Density 
fig, ax= plt.subplots(figsize=(15, 10))

# plot the particle density for M31 , 2D histogram
# ADD HERE
plt.hist2d(rn[:, 0], rn[:, 2], bins = 150, norm = LogNorm(), cmap = 'magma')

cbar = plt.colorbar()
cbar.set_label("Number of disk particle per bin", fontsize=15)

# Add axis labels
plt.xlabel('x (kpc)', fontsize=22)
plt.ylabel('z (kpc) ', fontsize=22)

#set axis limits
plt.ylim(-10,10)
plt.xlim(-45,45)

#adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

# Add countours
density_contour(rn[:, 0], rn[:, 2], 80, 80, ax=ax, colors=['red', 'yellow', 'white', 'yellow'])

# Save to a file
plt.savefig('Lab7_EdgeOn_Density.png')
# Rotated M31 Disk - FACE ON

# M31 Disk Density 
fig, ax= plt.subplots(figsize=(12, 10))

# plot the particle density for M31 
# ADD HERE
plt. hist2d(rn[:, 0], rn[:, 1], bins = 150, norm = LogNorm(), cmap = 'magma')

cbar = plt.colorbar()
cbar.set_label("Number of disk particle per bin", fontsize=15)

# make the contour plot
# x pos, y pos, contour res, contour res, axis, colors for contours.
# ADD HERE

# Add axis labels
plt.xlabel('x (kpc)', fontsize=22)
plt.ylabel('y (kpc)', fontsize=22)

#set axis limits
plt.ylim(-40,40)
plt.xlim(-40,40)

#adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

# Add countours
density_contour(rn[:, 0], rn[:, 1], 80, 80, ax=ax, colors=['red', 'yellow', 'white', 'yellow'])

# Save to a file 
plt.savefig('Lab7_FaceOn_Density.png')
# Plot velocity weighted EDGE ON DISK

fig = plt.figure(figsize=(15,10))
ax = plt.subplot(111)

# plot position of disk particles color 
# coded by velocity along the 3rd axis
# plt.scatter(pos1, pos2, c=vel1)
# ADD HERE 
plt.scatter(rn[:, 1], rn[:, 2], c = vn[:, 0])

#colorbar
cbar = plt.colorbar()
cbar.set_label('Vx (km/s) ', size=22)

# Add axis labels
plt.xlabel('y (kpc)', fontsize=22)
plt.ylabel('z (kpc)', fontsize=22)



#adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

#set axis limits
plt.ylim(-10,10)
plt.xlim(-40,40)

# Save file
plt.savefig('Lab7_EdgeOn_Vel.png')

# Create a mass profile object for M31 using homework solutions
M31 = MassProfile("M31", 0)
# arry of positions
rr = np.arange(0.01, 45, 0.1)
# Circular Velocity Profile
Vcirc = M31.circularVelocityTotal(rr)
# Make a phase diagram of the R vs V
# MW Disk Velocity Field edge on.

fig = plt.figure(figsize=(12,10))
ax = plt.subplot(111)

# Plot 2D Histogram for one component of  Pos vs Vel 
# ADD HERE
plt.hist2d(rn[:, 0], vn[:, 1], bins = 150, norm = LogNorm())
plt.colorbar()

# Overplot Circular Velocity from the MassProfile Code
# ADD HERE
plt.plot(rr, Vcirc, color = 'red')
plt.plot(-rr, -Vcirc, color = 'red')

# Add axis labels
plt.xlabel('x (kpc)', fontsize=22)
plt.ylabel('Vy (kpc) ', fontsize=22)



#adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size



# Save file
plt.savefig('Lab7_RotationCurve.png')
# Determine the positions of the disk particles in 
# cylindrical coordinates. (like in Lab 6)
cyl_r = np.sqrt(rn[:, 0] ** 2 + rn[:, 1] ** 2) # radial 
cyl_theta = np.arctan2(rn[:, 1], rn[:, 0]) * 180/np.pi # theta in degrees
# Make a phase diagram of R vs Theta

fig = plt.figure(figsize=(12,10))
ax = plt.subplot(111)

# Plot 2D Histogram of r vs theta
# ADD HERE
plt.hist2d(cyl_r, cyl_theta, bins = 150, norm = LogNorm())
plt.colorbar()




# Add axis labels
plt.xlabel('R (kpc) ', fontsize=22)
plt.ylabel(r'$\theta$ [deg] ', fontsize=22)



#adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

# Add countours
density_contour(cyl_r, cyl_theta, 80, 80, ax=ax, colors=['red', 'yellow', 'white', 'yellow'])

# Save file
plt.savefig('Lab7_SpiralPhase.png')
import os

html_file = "Lab7 Garg.html"
ipynb_file = "Lab7 Garg.ipynb"

# Delete old HTML if it exists
if os.path.exists(html_file):
    os.remove(html_file)

# Convert notebook to HTML
!jupyter nbconvert --to html "{ipynb_file}"


# ===== File: Lab8Garg.py =====
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



