"""
Code for part 4 taken from part 3 without any additional comments for simplicity
"""

import numpy as np
import astropy.units as u

def Read(filename):

    file = open(filename, 'r')
    line1 = file.readline()
    label, value = line1.split()
    
    time = float(value) * u.Myr
    line2 = file.readline()
    label, value = line2.split()
    
    total_particles = int(value)
    
    file.close()
    
    data = np.genfromtxt(filename, dtype=None, names=True, skip_header=3)
    
    return time, total_particles, data
