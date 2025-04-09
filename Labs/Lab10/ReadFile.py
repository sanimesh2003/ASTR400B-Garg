# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:35:49 2025

@author: kietb
"""

# Import Modules 
import numpy as np
import astropy.units as u

def Read(filename):
    '''
    This function will open and read the input file (MW_000.txt in this case).
    Inputs: 
        filename is the name of the input file
    Outputs: 
        the time, and total number of particles as variables 
        particle type, mass, x, y, z, vx, vy, vz columns as a data array
    '''
    # Open the file
    file = open(filename)

    # Read the first line and store the time in units of Myr
    line1 =  file.readline() # read the first line  
    label, value = line1.split() # split the line and assign it to label and value 
    time = float(value) * u.Myr # convert value into float and assign the unit Myr to it

    # Read the second line and store the total number of particles
    line2 = file.readline() # read the second line 
    label2, value2 = line2.split() # split the line and assign it to label2 and value2
    total_particles = int(value2) # convert to integer from text just in case

    # Close the file
    file.close()

    # Store the remainder of the file, matching them with the header information
    data = np.genfromtxt(filename, dtype = None, names = True, skip_header = 3)

    return time, total_particles, data 
