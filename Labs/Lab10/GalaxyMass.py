# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:07:02 2025

@author: kietb
"""

# Import modules 
import numpy as np
import astropy.units as u

from ReadFile import Read # import the Read function
# returning the time, and total number of particles as variables (don't need this)
# returning the data: particle type, mass, x, y, z, vx, vy, vz columns as a data array

def ComponentMass(filename, particle_type):
    ''' 
    This function will read the given file and return the total mass of any desired galaxy component, rounded to three decimal places. 
   Inputs: 
       filename is the input file
       particle_type is the type of particle
            This includes: type 1 (Halo), type 2 (Disk), and type 3 (Bulge) 
   Output: 
       total_mass (unit 10e12 M_sun) is the total mass of galaxy component
    '''

    # Retrieve values from the data file using Read function
    _, _, data = Read(filename) # ignoring the time and total_particles info

    # Checking indices for particles of a given type
    index = np.where(data['type'] == particle_type)

    # Take out the data of only the given type particles
    given_type_particles = data[index]

    # Extract the masses of particles (in 1e12 M_sun)
    mass = given_type_particles['m'] / 1e2 * u.M_sun # the original data given the mass in 10e10 M_sun

    # Add all the masses together, round to 3 decimal places
    total_mass = np.round(np.sum(mass),3)

    return total_mass

# Creating the data table for question 3
import pandas as pd

# Creating the dictionary to store galaxy informations
galaxies_table = {
    'Galaxy Name': ['MW', 'M31', 'M33'],
    'Halo Mass (10e12 M_sun)': [
        ComponentMass('MW_000.txt', 1).value, 
        ComponentMass('M31_000.txt', 1).value,
        ComponentMass('M33_000.txt', 1).value],
    'Disk Mass (10e12 M_sun)': [
        ComponentMass('MW_000.txt', 2).value, 
        ComponentMass('M31_000.txt', 2).value, 
        ComponentMass('M33_000.txt', 2).value],
    'Bulge Mass (10e12 M_sun)': [
        ComponentMass('MW_000.txt', 3).value, 
        ComponentMass('M31_000.txt', 3).value, 
        0.0] # no bulge at M33
}

# Convert the dictionary to Pandas dataframe
df = pd.DataFrame(galaxies_table)

# Compute total mass of each galaxy: 
df['Total Mass (10e12 M_sun)'] = df['Halo Mass (10e12 M_sun)'] + df['Disk Mass (10e12 M_sun)'] + df['Bulge Mass (10e12 M_sun)']

# Compute the bayron fraction for each galaxy and the whole local group 
# f_bar = total_stellar_mass / total_mass (dark + stellar)
df["f_bar"] = np.round((df["Disk Mass (10e12 M_sun)"] + df["Bulge Mass (10e12 M_sun)"]) / df["Total Mass (10e12 M_sun)"], 3)

# Compute the total mass of the Local Group and add it to the table
total_local_group_halo = df["Halo Mass (10e12 M_sun)"].sum()
total_local_group_disk = df["Disk Mass (10e12 M_sun)"].sum()
total_local_group_bulge = df["Bulge Mass (10e12 M_sun)"].sum()
total_local_group_mass = df["Total Mass (10e12 M_sun)"].sum()
total_local_group_fbar = np.round((total_local_group_disk + total_local_group_bulge) / total_local_group_mass, 3)
df.loc[len(df)] = ["Local Group", 
                   total_local_group_halo, 
                   total_local_group_disk,
                   total_local_group_bulge,
                   total_local_group_mass,
                   total_local_group_fbar]

# Display the table
print(df)
