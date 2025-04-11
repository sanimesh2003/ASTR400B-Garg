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

