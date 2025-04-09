# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:57:08 2025

@author: kietb
"""
# import modules
import numpy as np
import astropy.units as u
import astropy.table as tbl
import sys
import os

# Change path to homework3 where the ReadFile.py file is
module_path = r'C:\Users\kietb\OneDrive\Desktop\Suffering\Undergrad\ASTR400B\Homeworks\Homework3'

# Add the directory to sys.path
sys.path.append(module_path)

# Import ReadFile
from ReadFile import Read

class CenterOfMass:
# Class to define COM position and velocity properties 
# of a given galaxy and simulation snapshot

    def __init__(self, filename, ptype):
        ''' Class to calculate the 6-D phase-space position of a galaxy's center of mass using
        a specified particle type. 
            
            PARAMETERS
            ----------
            filename : `str`
                snapshot file
            ptype : `int; 1, 2, or 3`
                particle type to use for COM calculations
        '''
     
        # read data in the given file using Read
        self.time, self.total, self.data = Read(filename)                                                                                             

        #create an array to store indexes of particles of desired Ptype                                
        self.index = np.where(self.data['type'] == ptype)

        # store the mass, positions, velocities of only the particles of the given type
        # the following only gives the example of storing the mass
        self.m = self.data['m'][self.index] 
        # write your own code to complete this for positions and velocities
        self.x = self.data['x'][self.index]
        self.y = self.data['y'][self.index]
        self.z = self.data['z'][self.index]
        self.vx = self.data['vx'][self.index]
        self.vy = self.data['vy'][self.index]
        self.vz = self.data['vz'][self.index]


    def COMdefine(self,a,b,c,m):
        ''' Method to compute the COM of a generic vector quantity by direct weighted averaging.
        
        PARAMETERS
        ----------
        a : `float or np.ndarray of floats`
            first vector component
        b : `float or np.ndarray of floats`
            second vector component
        c : `float or np.ndarray of floats`
            third vector component
        m : `float or np.ndarray of floats`
            particle masses
        
        RETURNS
        -------
        a_com : `float`
            first component on the COM vector
        b_com : `float`
            second component on the COM vector
        c_com : `float`
            third component on the COM vector
        '''
        # write your own code to compute the generic COM 
        # using Eq. 1 in the homework instructions
        # x_COM = sum(xi + mi) / sum(mi)
        # xcomponent Center of mass
        a_com = np.sum(a * m) / np.sum(m)
        # ycomponent Center of mass
        b_com = np.sum(b * m) / np.sum(m)
        # zcomponent Center of mass
        c_com = np.sum(c * m) / np.sum(m)
        
        # return the 3 components separately
        return a_com, b_com, c_com
    
    
    def COM_P(self, delta, volDec):
        '''Method to compute the position of the center of mass of the galaxy 
        using the shrinking-sphere method.

        PARAMETERS
        ----------
        delta : `float`
            Error tolerance in kpc.
        volDec : `float`
            Factor by which the radius is divided during each iteration.

        RETURNS
        ----------
        p_COM : `np.ndarray of astropy.Quantity`
            3-D position of the center of mass in kpc.
        '''                                                                      
        # Center of Mass Position                                                                                      

        # Try a first guess at the COM position                                                                                       
        x_COM, y_COM, z_COM = self.COMdefine(self.x, self.y, self.z, self.m)
        r_COM = np.sqrt(x_COM ** 2 + y_COM ** 2 + z_COM ** 2)

        # Shift the reference frame to the guessed COM                                                                                 
        x_new = self.x - x_COM
        y_new = self.y - y_COM
        z_new = self.z - z_COM
        r_new = np.sqrt(x_new ** 2 + y_new ** 2 + z_new ** 2)

        # Maximum radius of the sphere                                                                                 
        r_max = max(r_new) / volDec
        
        # Initial change larger than delta                                                                             
        change = 1000.0
        
        # Iterative shrinking sphere method                                                                           
        while (change > delta):
            index2 = np.where(r_new < r_max)
            x2 = self.x[index2]
            y2 = self.y[index2]
            z2 = self.z[index2]
            m2 = self.m[index2]

            x_COM2, y_COM2, z_COM2 = self.COMdefine(x2, y2, z2, m2)
            r_COM2 = np.sqrt(x_COM2**2 + y_COM2**2 + z_COM2**2)

            change = np.abs(r_COM - r_COM2)

            # Reduce the radius by volDec instead of fixed 2.0
            r_max /= volDec

            x_new = self.x - x_COM2
            y_new = self.y - y_COM2
            z_new = self.z - z_COM2
            r_new = np.sqrt(x_new**2 + y_new**2 + z_new**2)

            x_COM, y_COM, z_COM = x_COM2, y_COM2, z_COM2
            r_COM = r_COM2

        p_COM = np.array([x_COM, y_COM, z_COM])

        return np.round(p_COM, 2) * u.kpc
                                                                   
        
    def COM_V(self, x_COM, y_COM, z_COM):
        ''' Method to compute the center of mass velocity based on the center of mass
        position.

        PARAMETERS
        ----------
        x_COM : 'astropy quantity'
            The x component of the center of mass in kpc
        y_COM : 'astropy quantity'
            The y component of the center of mass in kpc
        z_COM : 'astropy quantity'
            The z component of the center of mass in kpc
            
        RETURNS
        -------
        v_COM : `np.ndarray of astropy.Quantity'
            3-D velocity of the center of mass in km/s
        '''
        
        # the max distance from the center that we will use to determine 
        #the center of mass velocity                   
        rv_max = 15.0*u.kpc
        
        # determine the position of all particles relative to the center of mass position (x_COM, y_COM, z_COM)
        # write your own code below
        xV = self.x * u.kpc - x_COM
        yV = self.y * u.kpc - y_COM
        zV = self.z * u.kpc - z_COM
        rV = np.sqrt(xV ** 2 + yV ** 2 + zV ** 2)
        
        # determine the index for those particles within the max radius
        # write your own code below
        indexV = np.where(rV < rv_max)
        
        # determine the velocity and mass of those particles within the mas radius
        # write your own code below
        # Note that x_COM, y_COM, z_COM are astropy quantities and you can only subtract one astropy quantity from another
        # So, when determining the relative positions, assign the appropriate units to self.x
        vx_new = self.vx[indexV]
        vy_new = self.vy[indexV]
        vz_new = self.vz[indexV] 
        m_new =  self.m[indexV]
        
        # compute the center of mass velocity using those particles
        # write your own code below
        vx_COM, vy_COM, vz_COM = self.COMdefine(vx_new, vy_new, vz_new, m_new)
        
        # create an array to store the COM velocity
        # write your own code below
        v_COM = np.array([vx_COM, vy_COM, vz_COM])  * u.km / u.s

        # return the COM vector
        # set the correct units usint astropy
        # round all values                                                                                        
        return np.round(v_COM, 2)
