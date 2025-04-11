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

