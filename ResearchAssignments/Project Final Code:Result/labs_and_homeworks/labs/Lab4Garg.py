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

