import numpy as np
import astropy.units as u
import sys
import os

# Make sure you import the Read function (and possibly other needed packages) at the top
# Change path to homework3 where the ReadFile.py file is
module_path = r'C:\Users\kietb\OneDrive\Desktop\Suffering\Undergrad\ASTR400B\Homeworks\Homework3'

# Add the directory to sys.path
sys.path.append(module_path)

# Import ReadFile
from ReadFile import Read

class CenterOfMass:
    # Class to define COM position and velocity properties of a given galaxy and simulation snapshot
    
    def __init__(self, filename, ptype):
        ''' 
        Class to calculate the 6-D phase-space center of mass of a galaxy 
        using a specified particle type.
        
        PARAMETERS
        ----------
        filename : str
            Snapshot file name (e.g., 'MW_000.txt')
        ptype : int
            Particle type (1=Halo, 2=Disk, 3=Bulge) for which to compute COM
        '''
     
        # 1) Read data in the given file using Read
        self.time, self.total, self.data = Read(filename)  
        
        # 2) Create an index array to store indices of particles of the desired Ptype
        self.index = np.where(self.data['type'] == ptype)
        
        # 3) Store the mass, positions, velocities of only the particles of the given type
        self.m = self.data['m'][self.index]          # masses
        self.x = self.data['x'][self.index]          # x-positions
        self.y = self.data['y'][self.index]          # y-positions
        self.z = self.data['z'][self.index]          # z-positions
        self.vx = self.data['vx'][self.index]        # x-velocities
        self.vy = self.data['vy'][self.index]        # y-velocities
        self.vz = self.data['vz'][self.index]        # z-velocities


    def COMdefine(self, a, b, c, m):
        '''
        Method to compute the generic center of mass (COM) of a given vector
        quantity (e.g., position or velocity) by direct weighted averaging.
        
        PARAMETERS
        ----------
        a : float or np.ndarray
            first component array (e.g., x or vx)
        b : float or np.ndarray
            second component array (e.g., y or vy)
        c : float or np.ndarray
            third component array (e.g., z or vz)
        m : float or np.ndarray
            array of particle masses
        
        RETURNS
        -------
        a_com : float
            COM of the first component
        b_com : float
            COM of the second component
        c_com : float
            COM of the third component
        '''
        
        # Weighted sum in each dimension
        a_com = np.sum(a * m) / np.sum(m)
        b_com = np.sum(b * m) / np.sum(m)
        c_com = np.sum(c * m) / np.sum(m)

        return a_com, b_com, c_com


    def COM_P(self, delta=0.1):
        '''
        Method to compute the position of the center of mass of the galaxy 
        using the shrinking-sphere method, iterating until convergence.
        
        PARAMETERS
        ----------
        delta : float, optional
            Error tolerance in kpc for stopping criterion. Default = 0.1 kpc
        
        RETURNS
        -------
        p_COM : np.ndarray of astropy.Quantity
            3-D position of the center of mass in kpc (rounded to 2 decimals)
        '''
        
        # 1) First guess at COM position using all particles
        x_COM, y_COM, z_COM = self.COMdefine(self.x, self.y, self.z, self.m)
        
        # 2) Compute the magnitude of the COM position vector
        r_COM = np.sqrt(x_COM**2 + y_COM**2 + z_COM**2)
        
        # 3) Shift to COM frame (for the *initial* guess)
        x_new = self.x - x_COM
        y_new = self.y - y_COM
        z_new = self.z - z_COM
        r_new = np.sqrt(x_new**2 + y_new**2 + z_new**2)
        
        # 4) Find the maximum 3D distance from this COM, then halve it
        r_max = np.max(r_new) / 2.0
        
        # 5) Set an initial change to be very large (so loop starts)
        change = 1000.0
        
        # 6) Iteratively refine until the COM position changes by less than delta
        while (change > delta):
            
            # Select particles within the reduced radius from the ORIGINAL positions,
            # but recentered around the last COM guess
            x_new = self.x - x_COM
            y_new = self.y - y_COM
            z_new = self.z - z_COM
            r_new = np.sqrt(x_new**2 + y_new**2 + z_new**2)
            
            index2 = np.where(r_new < r_max)
            
            # Retrieve only those particles
            x2 = self.x[index2]
            y2 = self.y[index2]
            z2 = self.z[index2]
            m2 = self.m[index2]
            
            # Recompute COM with these "in-sphere" particles
            x_COM2, y_COM2, z_COM2 = self.COMdefine(x2, y2, z2, m2)
            r_COM2 = np.sqrt(x_COM2**2 + y_COM2**2 + z_COM2**2)
            
            # Check how much COM changed from previous iteration
            change = np.abs(r_COM - r_COM2)
            # print("CHANGE = ", change)
            
            # Halve the radius again for the next iteration
            r_max /= 2.0
            
            # Reset COM values to the newly computed values for next loop
            x_COM = x_COM2
            y_COM = y_COM2
            z_COM = z_COM2
            r_COM = r_COM2
        
        # Once convergence is reached:
        p_COM = np.array([x_COM, y_COM, z_COM]) * u.kpc
        # Round to 2 decimal places
        p_COM = np.round(p_COM, 2)
        
        return p_COM


    def COM_V(self, x_COM, y_COM, z_COM):
        '''
        Method to compute the center of mass velocity based on the center of mass position.
        
        PARAMETERS
        ----------
        x_COM : astropy.Quantity
            The x component of the COM in kpc
        y_COM : astropy.Quantity
            The y component of the COM in kpc
        z_COM : astropy.Quantity
            The z component of the COM in kpc
            
        RETURNS
        -------
        v_COM : np.ndarray of astropy.Quantity
            3-D velocity of the center of mass in km/s (rounded to 2 decimals)
        '''
        
        # Maximum distance from the center to consider when computing COM velocity
        rv_max = 15.0 * u.kpc
        
        # Convert COM to "raw" floats if needed
        # (Assuming self.x etc. are in kpc, we can handle them directly)
        xC = x_COM.value
        yC = y_COM.value
        zC = z_COM.value
        
        # Determine positions relative to the COM
        xV = self.x - xC
        yV = self.y - yC
        zV = self.z - zC
        
        # 3D distance of each particle from COM
        rV = np.sqrt(xV**2 + yV**2 + zV**2)
        
        # Select those particles within rv_max
        indexV = np.where(rV < rv_max.value)
        
        # Retrieve velocities for those particles
        vx_new = self.vx[indexV]
        vy_new = self.vy[indexV]
        vz_new = self.vz[indexV]
        m_new  = self.m[indexV]
        
        # Compute COM velocity
        vx_COM, vy_COM, vz_COM = self.COMdefine(vx_new, vy_new, vz_new, m_new)
        
        # Create an array for the COM velocity and convert to astropy with km/s
        v_COM = np.array([vx_COM, vy_COM, vz_COM]) * u.km/u.s
        # Round to 2 decimal places
        v_COM = np.round(v_COM, 2)
        
        return v_COM