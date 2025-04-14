import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import os
import pandas as pd
import glob
from scipy.optimize import curve_fit
    
class CenterOfMass:



    def __init__(self, data, ptype):
        """
        PARAMETERS
        ----------
        data   : np.ndarray
            Structured array from our Read() function, containing columns
            'type', 'm', 'x', 'y', 'z', 'vx', 'vy', 'vz'
        ptype  : int
            The particle type of interest (1=Halo, 2=Disk, 3=Bulge)
        """

        idx = np.where(data['type'] == ptype)
        
        self.m  = data['m'][idx]     # array of masses
        self.x  = data['x'][idx]    
        self.y  = data['y'][idx]    
        self.z  = data['z'][idx]    
        self.vx = data['vx'][idx]    
        self.vy = data['vy'][idx]    
        self.vz = data['vz'][idx]    

    def COMdefine(self, a, b, c, m):
        """
        Helper function to compute the (mass-weighted) average of
        any 3D component vectors (like x,y,z) or (vx,vy,vz).
        
        PARAMETERS
        ----------
        a, b, c : array-like
            Coordinates or velocities in each dimension
        m       : array-like
            Corresponding mass array
        RETURNS
        -------
        a_com, b_com, c_com : float
            Weighted average in each dimension
        """
        a_com = np.sum(a * m) / np.sum(m)
        b_com = np.sum(b * m) / np.sum(m)
        c_com = np.sum(c * m) / np.sum(m)
        return a_com, b_com, c_com

    def COM_P(self, delta=0.1):
        """
        Iterative method to determine the center of mass position
        using a shrinking-sphere approach.

        PARAMETERS
        ----------
        delta : float
            Convergence tolerance in the distance of subsequent COM estimates (kpc)

        RETURNS
        -------
        (xCOM, yCOM, zCOM) : tuple of floats
            The final COM position in (kpc).
        """
        # 1) Initial guess of COM using all particles
        xCOM, yCOM, zCOM = self.COMdefine(self.x, self.y, self.z, self.m)
        rCOM = np.sqrt(xCOM**2 + yCOM**2 + zCOM**2)

        # 2) recenter the positions about this COM guess
        x_new = self.x - xCOM
        y_new = self.y - yCOM
        z_new = self.z - zCOM
        # distances of each particle from the COM
        r_new = np.sqrt(x_new**2 + y_new**2 + z_new**2)

        # 3) find the maximum distance and then half it
        r_max = np.max(r_new) / 2.0

        # define a large change so we start iterating
        change = 1000.0

        # 4) loop until the COM changes by less than delta
        while (change > delta):
            # select those within the reduced radius
            idx_within = np.where(r_new < r_max)[0]
            # compute new COM
            x2, y2, z2 = self.COMdefine(x_new[idx_within],
                                        y_new[idx_within],
                                        z_new[idx_within],
                                        self.m[idx_within])
            r2 = np.sqrt(x2**2 + y2**2 + z2**2)

            change = np.abs(rCOM - r2)

            xCOM += x2
            yCOM += y2
            zCOM += z2

            rCOM = r2
            # recenter all particles
            x_new = self.x - xCOM
            y_new = self.y - yCOM
            z_new = self.z - zCOM
            r_new = np.sqrt(x_new**2 + y_new**2 + z_new**2)

            r_max /= 2.0

        return xCOM, yCOM, zCOM

    def COM_V(self, xCOM, yCOM, zCOM, rvmax=15.0):
        """
        Compute the center of mass velocity by selecting all
        particles within a chosen radius (rvmax, default=15 kpc)
        around the already-known COM position.

        PARAMETERS
        ----------
        xCOM, yCOM, zCOM : floats
            Known center-of-mass position for the galaxy
        rvmax : float
            The radius (kpc) within which to calculate velocities

        RETURNS
        -------
        (vxCOM, vyCOM, vzCOM) : tuple of floats
            The center-of-mass velocity in km/s
        """
        # distances from that COM
        dx = self.x - xCOM
        dy = self.y - yCOM
        dz = self.z - zCOM
        rr = np.sqrt(dx**2 + dy**2 + dz**2)
        # select those within rvmax
        idx = np.where(rr < rvmax)[0]
        # compute COM velocity
        vxCOM, vyCOM, vzCOM = self.COMdefine(self.vx[idx],
                                             self.vy[idx],
                                             self.vz[idx],
                                             self.m[idx])
        return vxCOM, vyCOM, vzCOM
