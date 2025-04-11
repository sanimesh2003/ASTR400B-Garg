# ===== File: HW2ParticleProperties.py =====
import numpy as np  # For numerical operations
import astropy.units as u  # For handling units (kpc, km/s, M_sun)
from ReadFile import Read  # Import the `Read` function from `ReadFile.py`
"""
ParticleProperties.py
This script defines the `ParticleInfo` function, which calculates properties of a specific particle
(distance, velocity, and mass) for a given particle type and number, using data read by `ReadFile`.
"""

# Define the ParticleInfo function
def ParticleInfo(filename, particle_type, particle_number):
    """
    Function to calculate the distance, velocity, and mass of a specific particle.

    Parameters:
        filename (str): Path to the data file (e.g., MW_000.txt).
        particle_type (int): Particle type (1 = Dark Matter, 2 = Disk Stars, 3 = Bulge Stars).
        particle_number (int): Index of the particle within the specified type.

    Returns:
        distance (float): Magnitude of the 3D distance in kpc, rounded to 3 decimal places.
        velocity (float): Magnitude of the 3D velocity in km/s, rounded to 3 decimal places.
        mass (float): Mass of the particle in units of solar mass (M_sun).
    """

    time, total_particles, data = Read(filename)
    """
    The "Read" function returns:
    - "time": Time in Myr from the first row of the file.
    - "total_particles": Total number of particles in the file.
    - "data": A structured array with columns for particle properties (type, mass, position, velocity).
    """

    index = np.where(data['type'] == particle_type)  # Filter data to match the given particle type
    """
    The "np.where" function returns the indices of rows in "data" where 'type' matches the "particle_type".
    For example, if particle_type = 2 (Disk Stars), "index" contains the indices of all disk stars.
    """

    # Extract the position (x, y, z) of the specified particle and assign units of kpc
    x = data['x'][index][particle_number] * u.kpc  # x-coordinate of the particle
    y = data['y'][index][particle_number] * u.kpc  # y-coordinate of the particle
    z = data['z'][index][particle_number] * u.kpc  # z-coordinate of the particle

    # Extract the velocity (vx, vy, vz) of the specified particle and assign units of km/s
    vx = data['vx'][index][particle_number] * u.km / u.s  # x-component of velocity
    vy = data['vy'][index][particle_number] * u.km / u.s  # y-component of velocity
    vz = data['vz'][index][particle_number] * u.km / u.s  # z-component of velocity

    # Extract the mass of the specified particle and convert it to units of solar mass (M_sun)
    mass = data['m'][index][particle_number] * 1e10 * u.M_sun  # Mass in solar masses
    """
    The mass values in the file are stored in units of 10^10 M_sun. To convert them to solar masses,
    multiply by 10^10 and assign the unit `u.M_sun`.
    """

    # Calculate the magnitude of the distance (3D distance) in kpc
    distance = np.sqrt(x**2 + y**2 + z**2).to(u.kpc).value  # Convert to kpc and extract the numeric value

    # Calculate the magnitude of the velocity (3D velocity) in km/s
    velocity = np.sqrt(vx**2 + vy**2 + vz**2).to(u.km / u.s).value  # Convert to km/s and extract the numeric value

    # Round the distance and velocity to 3 decimal places
    distance = np.around(distance, 3)
    velocity = np.around(velocity, 3)

    # Return the computed properties
    return distance, velocity, mass.value

# Testing the function
if __name__ == "__main__":
    """
    Test block to verify the functionality of the `ParticleInfo` function.
    This block will not be executed when the script is imported elsewhere.
    """
    # Example particle specifications
    filename = "MW_000.txt"  
    particle_type = 2  # Disk Stars
    particle_number = 0  # First particle of the specified type

    # Call the function to calculate particle properties
    distance, velocity, mass = ParticleInfo(filename, particle_type, particle_number)

    # Print the results for verification
    print("Distance (kpc):", distance)  # 3D distance of the particle in kpc
    print("Velocity (km/s):", velocity)  # 3D velocity of the particle in km/s
    print("Mass (M_sun):", mass)  # Mass of the particle in solar masses
# Part 5 of the Homework - Done within the script here itself

# Define the parameters for the particle
filename = "MW_000.txt"  # File containing particle data (replace with the correct path if needed)
particle_type = 2  # Particle type: 2 corresponds to Disk Stars
particle_number = 99  # Index of the 100th particle (indexing starts from 0)

# Get the properties of the specified particle
distance, velocity, mass = ParticleInfo(filename, particle_type, particle_number)
"""
The "ParticleInfo" function calculates:
- "distance": The 3D distance of the particle in kpc.
- "velocity": The 3D velocity of the particle in km/s.
- "mass": The mass of the particle in solar masses (M_sun).
"""

# Convert the 3D distance from kpc to light-years
distance_ly = (distance * u.kpc).to(u.lyr).value  # Convert from kiloparsecs to light-years
distance_ly = np.around(distance_ly, 3)  # Round the value to 3 decimal places for clarity

# Print the calculated properties of the particle
print(f"3D Distance (kpc): {distance}")  # Print the distance in kpc
print(f"3D Velocity (km/s): {velocity}")  # Print the velocity in km/s
print(f"Mass (M_sun): {mass}")  # Print the mass in solar masses
print(f"3D Distance (light-years): {distance_ly}")  # Print the distance in light-years




# ===== File: HW3GalaxyMass.py =====
"""
GalaxyMass.py
This script contains the `ComponentMass` function, which calculates the total mass 
of a specified galaxy component (Halo, Disk, or Bulge) using data from a file.
"""

# Import necessary modules
import numpy as np  # For numerical calculations
from ReadFile import Read  # Import Read function from Homework 2

def ComponentMass(filename, particle_type):
    """
    Function to compute the total mass of a given galaxy component.

    Parameters:
        filename (str): The name of the data file (e.g., "MW_000.txt").
        particle_type (int): The type of particle (1 = Halo, 2 = Disk, 3 = Bulge).

    Returns:
        float: Total mass of the component in units of 10^12 solar masses, rounded to 3 decimal places.
    """
    # Read the data file
    time, total_particles, data = Read(filename)

    # Filter particles of the given type
    index = np.where(data['type'] == particle_type)  # Get indices for the selected particle type
    mass = np.sum(data['m'][index])  # Sum the mass of selected particles

    # Convert mass from 10^10 solar masses to 10^12 solar masses
    mass = mass * 1e-2  

    # Round to 3 decimal places
    return np.round(mass, 3)

# Testing the function
if __name__ == "__main__":
    filename = "MW_000.txt"  # Data file
    halo_mass = ComponentMass(filename, 1)  # Compute Halo mass
    disk_mass = ComponentMass(filename, 2)  # Compute Disk mass
    bulge_mass = ComponentMass(filename, 3)  # Compute Bulge mass

    # Print the results
    print(f"Halo Mass (10^12 solar masses): {halo_mass}")
    print(f"Disk Mass (10^12 solar masses): {disk_mass}")
    print(f"Bulge Mass (10^12 solar masses): {bulge_mass}")

"""
LocalGroupMass.py
This script calculates the mass breakdown of the Local Group galaxies (MW, M31, M33).
It organizes the data into a table and computes fbar for each galaxy and the entire Local Group.
"""

# Import necessary modules
import numpy as np  # For numerical calculations
import pandas as pd  # For organizing data into tables
from GalaxyMass import ComponentMass  # Import the ComponentMass function

# Define the filenames for each galaxy
galaxies = {
    "MW": "MW_000.txt",
    "M31": "M31_000.txt",
    "M33": "M33_000.txt"
}

# Initialize an empty list to store mass results
mass_results = []

# Loop through each galaxy and compute the masses
for galaxy, filename in galaxies.items():
    halo_mass = ComponentMass(filename, 1)  # Halo mass
    disk_mass = ComponentMass(filename, 2)  # Disk mass

    # M33 does not have a bulge, so set bulge mass to 0
    bulge_mass = ComponentMass(filename, 3) if galaxy != "M33" else 0

    # Compute the total mass of the galaxy
    total_mass = halo_mass + disk_mass + bulge_mass

    # Compute baryon fraction fbar
    fbar = (disk_mass + bulge_mass) / total_mass

    # Store results in the list
    mass_results.append([galaxy, halo_mass, disk_mass, bulge_mass, total_mass, np.round(fbar, 3)])

# Convert the results into a Pandas DataFrame
columns = ["Galaxy", "Halo Mass (10^12)", "Disk Mass (10^12)", "Bulge Mass (10^12)", "Total Mass (10^12)", "fbar"]
mass_table = pd.DataFrame(mass_results, columns=columns)

# Compute the total mass of the Local Group
local_group_mass = mass_table["Total Mass (10^12)"].sum()

# Compute the Local Group baryon fraction
total_stellar_mass = mass_table["Disk Mass (10^12)"].sum() + mass_table["Bulge Mass (10^12)"].sum()
fbar_local_group = total_stellar_mass / local_group_mass

# Append Local Group totals to the table
mass_table.loc[len(mass_table.index)] = ["Local Group", "-", "-", "-", local_group_mass, np.round(fbar_local_group, 3)]

# Display the table
print(mass_table)

# Save the table as a PDF
mass_table.to_csv("LocalGroupMass.csv", index=False)  # Save as CSV first

# Convert CSV to PDF using LaTeX (Bonus Points), 
# I would do this directly in python but for whatever reason I can't get mactext to install correctly
latex_code = mass_table.to_latex(index=False)

with open("LocalGroupMass.tex", "w") as f:
    f.write(latex_code)

print("Table saved as LocalGroupMass.csv and LocalGroupMass.tex")




# ===== File: HW4CenterOfMass.py =====
# Homework 4
# Center of Mass Position and Velocity
# Animesh Garg
# remember this is just a template,
# you don't need to follow every step.
# If you have your own method to solve the homework,
# it is totally fine
# import modules
import numpy as np
import astropy.units as u
import astropy.table as tbl

from ReadFile import Read
import numpy as np
import astropy.units as u

# Make sure you import the Read function (and possibly other needed packages) at the top
# from ReadFile import Read

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

# Create a Center of Mass object for the MW, M31, and M33.
# Example using particle type 2 (disk).

MW_COM = CenterOfMass("MW_000.txt", 2)
M31_COM = CenterOfMass("M31_000.txt", 2)
M33_COM = CenterOfMass("M33_000.txt", 2)

# Compute and print the COM position for each galaxy 
# using a tolerance of 0.1 kpc.

MW_COM_p = MW_COM.COM_P(0.1)
print("MW COM Position:", MW_COM_p)

M31_COM_p = M31_COM.COM_P(0.1)
print("M31 COM Position:", M31_COM_p)

M33_COM_p = M33_COM.COM_P(0.1)
print("M33 COM Position:", M33_COM_p)

# Compute and print the COM velocity for each galaxy 
# based on the COM position previously calculated.

MW_COM_v = MW_COM.COM_V(MW_COM_p[0], MW_COM_p[1], MW_COM_p[2])
print("MW COM Velocity:", MW_COM_v)

M31_COM_v = M31_COM.COM_V(M31_COM_p[0], M31_COM_p[1], M31_COM_p[2])
print("M31 COM Velocity:", M31_COM_v)

M33_COM_v = M33_COM.COM_V(M33_COM_p[0], M33_COM_p[1], M33_COM_p[2])
print("M33 COM Velocity:", M33_COM_v)

# (Optional) Additional cell: compute separations and relative velocities

# Separation between MW and M31
diff_pos_MW_M31 = MW_COM_p - M31_COM_p
sep_MW_M31 = np.sqrt(diff_pos_MW_M31[0]**2 + diff_pos_MW_M31[1]**2 + diff_pos_MW_M31[2]**2)
print(f"Separation MW-M31 (kpc): {sep_MW_M31:.3f}")

# Relative velocity between MW and M31
diff_vel_MW_M31 = MW_COM_v - M31_COM_v
rel_vel_MW_M31 = np.sqrt(diff_vel_MW_M31[0]**2 + diff_vel_MW_M31[1]**2 + diff_vel_MW_M31[2]**2)
print(f"Relative velocity MW-M31 (km/s): {rel_vel_MW_M31:.3f}")

# Repeat for M31-M33
diff_pos_M31_M33 = M31_COM_p - M33_COM_p
sep_M31_M33 = np.sqrt(diff_pos_M31_M33[0]**2 + diff_pos_M31_M33[1]**2 + diff_pos_M31_M33[2]**2)
print(f"Separation M31-M33 (kpc): {sep_M31_M33:.3f}")

diff_vel_M31_M33 = M31_COM_v - M33_COM_v
rel_vel_M31_M33 = np.sqrt(diff_vel_M31_M33[0]**2 + diff_vel_M31_M33[1]**2 + diff_vel_M31_M33[2]**2)
print(f"Relative velocity M31-M33 (km/s): {rel_vel_M31_M33:.3f}")




# ===== File: HW5MassProfile.py =====
import numpy as np
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt

# Import your Read and CenterOfMass functions
from ReadFile import Read
from CenterOfMass import CenterOfMass

class MassProfile:
    """
    A class to calculate mass profiles and rotation curves for a given galaxy at a given snapshot.
    """

    def __init__(self, galaxy, snap):
        """
        Initialize the class by reading in data from a snapshot file, and 
        computing the center of mass from disk particles.

        Parameters
        ----------
        galaxy : str
            The name of the galaxy ('MW', 'M31', or 'M33').
        snap : int
            The snapshot number, e.g., 0 for present day.
        """

        # Store galaxy name as a property
        self.gname = galaxy
        
        # Construct the filename. Example: 'MW_000.txt' if galaxy='MW' and snap=0
        ilbl = '000' + str(snap)
        ilbl = ilbl[-3:]
        self.filename = f"{galaxy}_{ilbl}.txt"
        
        # Read the data from this file
        self.time, self.total, self.data = Read(self.filename)
        
        # Store positions and masses in astropy units
        # NOTE: We multiply the mass by 1e10 Msun based on typical usage in these snapshots
        self.x = self.data['x'] * u.kpc
        self.y = self.data['y'] * u.kpc
        self.z = self.data['z'] * u.kpc
        self.m = self.data['m'] * 1e10 * u.Msun  # Msun
        
        # Compute the galaxy's COM position using DISK particles
        com_object = CenterOfMass(self.filename, 2)  # 2 = disk
        self.com_pos = com_object.COM_P(0.1)  # returns [x_COM, y_COM, z_COM] in kpc
        
        # Store G in convenient units: kpc (km/s)^2 / Msun
        self.G = G.to(u.kpc * u.km**2 / (u.s**2 * u.Msun))


    def MassEnclosed(self, ptype, radii):
        """
        Computes the mass enclosed within each radius in 'radii' (array) for particles of type 'ptype'.
        
        Parameters
        ----------
        ptype : int
            Particle type: 1 (Halo), 2 (Disk), 3 (Bulge).
        radii : np.ndarray or list of floats
            Radii in kpc at which to compute enclosed mass.
        
        Returns
        -------
        enclosed_mass : astropy.units.Quantity
            Array of enclosed masses at each radius in Msun.
        """
        # Select only particles of the requested type
        index = np.where(self.data['type'] == ptype)
        m_selected = self.m[index]  # Msun
        x_selected = self.x[index]  # kpc
        y_selected = self.y[index]  # kpc
        z_selected = self.z[index]  # kpc
        
        # Center relative to COM
        dx = x_selected - self.com_pos[0]
        dy = y_selected - self.com_pos[1]
        dz = z_selected - self.com_pos[2]
        r_part = np.sqrt(dx**2 + dy**2 + dz**2)  # distance of each particle from COM
        
        # Ensure 'radii' is an astropy quantity in kpc
        radii_kpc = radii * u.kpc
        
        # Initialize array for enclosed mass
        enclosed_mass = np.zeros(len(radii_kpc)) * u.Msun
        
        # Loop over the input radii
        for i in range(len(radii_kpc)):
            index_within = np.where(r_part < radii_kpc[i])
            enclosed_mass[i] = np.sum(m_selected[index_within])
        
        return enclosed_mass


    def MassEnclosedTotal(self, radii):
        """
        Computes the total enclosed mass (halo+disk+bulge) for each radius in 'radii'.

        Parameters
        ----------
        radii : array-like
            Radii in kpc.

        Returns
        -------
        total_mass : astropy.units.Quantity
            Array of total enclosed mass at each radius in Msun.
        """
        # Halo + Disk
        m_halo = self.MassEnclosed(1, radii)
        m_disk = self.MassEnclosed(2, radii)
        
        # Bulge (unless M33, which has no bulge)
        if self.gname == 'M33':
            m_bulge = 0 * m_halo
        else:
            m_bulge = self.MassEnclosed(3, radii)
        
        total_mass = m_halo + m_disk + m_bulge
        return total_mass


    def HernquistMass(self, r, a, Mhalo):
        """
        Computes the Hernquist 1990 mass profile: M(r) = Mhalo * (r^2 / (r+a)^2).
        
        Parameters
        ----------
        r : float or np.ndarray
            Radius (kpc)
        a : float
            Scale radius (kpc)
        Mhalo : float
            Total halo mass in Msun
        
        Returns
        -------
        M_hern : astropy.units.Quantity
            Enclosed mass at radius r in Msun.
        """
        r_kpc = r * u.kpc
        a_kpc = a * u.kpc
        Mhalo_msun = Mhalo * u.Msun
        
        frac = (r_kpc**2) / (r_kpc + a_kpc)**2
        M_hern = Mhalo_msun * frac
        return M_hern


    def CircularVelocity(self, ptype, radii):
        """
        Computes the circular velocity for the given component at each radius in 'radii'.
        
        Vc(r) = sqrt(G * M_enclosed(r) / r)
        
        Parameters
        ----------
        ptype : int
            Particle type (1=Halo, 2=Disk, 3=Bulge).
        radii : np.ndarray
            Radii in kpc.
        
        Returns
        -------
        V_circ : astropy.units.Quantity
            Circular velocity in km/s (rounded to 2 decimals).
        """
        # Enclosed mass
        Menc = self.MassEnclosed(ptype, radii)  # Msun
        r_kpc = radii * u.kpc
        
        V_circ = np.sqrt(self.G * Menc / r_kpc)
        return np.round(V_circ, 2)


    def CircularVelocityTotal(self, radii):
        """
        Computes the total circular velocity (halo + disk + bulge) at each radius in 'radii'.
        
        V_circ_total(r) = sqrt(G * [M_halo + M_disk + M_bulge] / r)
        
        Parameters
        ----------
        radii : array-like
            Radii in kpc.
        
        Returns
        -------
        V_circ_total : astropy.units.Quantity
            Total circular velocity in km/s (rounded to 2 decimals).
        """
        Mtot = self.MassEnclosedTotal(radii)  # Msun
        r_kpc = radii * u.kpc
        
        V_circ = np.sqrt(self.G * Mtot / r_kpc)
        return np.round(V_circ, 2)


    def HernquistVCirc(self, r, a, Mhalo):
        """
        Computes the circular velocity for a Hernquist halo:
        M(r) = Mhalo * (r^2 / (r+a)^2).
        V_circ(r) = sqrt(G * M(r) / r).
        
        Parameters
        ----------
        r : float or array-like
            Radius in kpc.
        a : float
            Scale radius in kpc.
        Mhalo : float
            Halo mass in Msun.
        
        Returns
        -------
        V_hern : astropy.units.Quantity
            Hernquist circular velocity in km/s (rounded to 2 decimals).
        """
        r_kpc = r * u.kpc
        a_kpc = a * u.kpc
        Mhalo_msun = Mhalo * u.Msun
        
        # M(r) for Hernquist
        M_hern = Mhalo_msun * (r_kpc**2 / (r_kpc + a_kpc)**2)
        
        V_hern = np.sqrt(self.G * M_hern / r_kpc)
        return np.round(V_hern, 2)

# Create MassProfile objects for each galaxy at Snapshot 0
MW = MassProfile("MW", 0)
M31 = MassProfile("M31", 0)
M33 = MassProfile("M33", 0)

# We often plot from 0.1 to 30 kpc
r_plot = np.linspace(0.1, 30, 1000)

##################################
# 1) Milky Way Mass Profile
##################################
# Enclosed mass of each component up to 30 kpc
M_halo_MW_30 = MW.MassEnclosed(1, r_plot)
M_disk_MW_30 = MW.MassEnclosed(2, r_plot)
M_bulge_MW_30 = MW.MassEnclosed(3, r_plot)
M_total_MW_30 = MW.MassEnclosedTotal(r_plot)

# Use a bigger radius for the total halo mass
big_radius = 300.0  # kpc
# Compute halo mass at that radius:
M_halo_MW_300 = MW.MassEnclosed(1, np.array([big_radius]))
MW_halo_mass_total = M_halo_MW_300[0].value  # a float in Msun

# Choose a Hernquist scale radius (trial & error):
a_guess_MW = 60.0  # kpc
# Get the theoretical Hernquist halo mass at 0.1-30 kpc
M_hern_MW_30 = MW.HernquistMass(r_plot, a_guess_MW, MW_halo_mass_total)

# Plot for MW
plt.figure(figsize=(8,6))
plt.semilogy(r_plot, M_halo_MW_30, color='b', label='Halo')
plt.semilogy(r_plot, M_disk_MW_30, color='r', label='Disk')
plt.semilogy(r_plot, M_bulge_MW_30, color='g', label='Bulge')
plt.semilogy(r_plot, M_total_MW_30, color='k', label='Total')

# Overplot Hernquist
plt.semilogy(r_plot, M_hern_MW_30, 'b--',
             label=f'Hernquist Halo (a={a_guess_MW} kpc)\nMhalo={MW_halo_mass_total:.2e} Msun')

plt.title('1) Mass Profile: Milky Way', fontsize=15)
plt.xlabel('r (kpc)', fontsize=14)
plt.ylabel('Enclosed Mass (Msun)', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.savefig("1 Mass Profile Milky Way.png", dpi=300, bbox_inches="tight")
plt.show()


##################################
# 2) M31 Mass Profile
##################################
M_halo_M31_30 = M31.MassEnclosed(1, r_plot)
M_disk_M31_30 = M31.MassEnclosed(2, r_plot)
M_bulge_M31_30 = M31.MassEnclosed(3, r_plot)
M_total_M31_30 = M31.MassEnclosedTotal(r_plot)

# Use big radius for total halo mass
M_halo_M31_300 = M31.MassEnclosed(1, np.array([big_radius]))
M31_halo_mass_total = M_halo_M31_300[0].value

a_guess_M31 = 60.0  # kpc
M_hern_M31_30 = M31.HernquistMass(r_plot, a_guess_M31, M31_halo_mass_total)

plt.figure(figsize=(8,6))
plt.semilogy(r_plot, M_halo_M31_30, color='b', label='Halo')
plt.semilogy(r_plot, M_disk_M31_30, color='r', label='Disk')
plt.semilogy(r_plot, M_bulge_M31_30, color='g', label='Bulge')
plt.semilogy(r_plot, M_total_M31_30, color='k', label='Total')

plt.semilogy(r_plot, M_hern_M31_30, 'b--',
             label=f'Hernquist Halo (a={a_guess_M31} kpc)\nMhalo={M31_halo_mass_total:.2e} Msun')

plt.title('2) Mass Profile: M31', fontsize=15)
plt.xlabel('r (kpc)', fontsize=14)
plt.ylabel('Enclosed Mass (Msun)', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.savefig("2 Mass Profile M31.png", dpi=300, bbox_inches="tight")
plt.show()


##################################
# 3) M33 Mass Profile
##################################
M_halo_M33_30 = M33.MassEnclosed(1, r_plot)
M_disk_M33_30 = M33.MassEnclosed(2, r_plot)
# M33 has no bulge, so bulge mass is zero
M_bulge_M33_30 = 0.0 * M_halo_M33_30
M_total_M33_30 = M33.MassEnclosedTotal(r_plot)

# Use big radius for total halo mass
M_halo_M33_300 = M33.MassEnclosed(1, np.array([big_radius]))
M33_halo_mass_total = M_halo_M33_300[0].value

a_guess_M33 = 25.0  # kpc
M_hern_M33_30 = M33.HernquistMass(r_plot, a_guess_M33, M33_halo_mass_total)

plt.figure(figsize=(8,6))
plt.semilogy(r_plot, M_halo_M33_30, color='b', label='Halo')
plt.semilogy(r_plot, M_disk_M33_30, color='r', label='Disk')
plt.semilogy(r_plot, M_bulge_M33_30, color='g', label='Bulge (none)')
plt.semilogy(r_plot, M_total_M33_30, color='k', label='Total')

plt.semilogy(r_plot, M_hern_M33_30, 'b--',
             label=f'Hernquist Halo (a={a_guess_M33} kpc)\nMhalo={M33_halo_mass_total:.2e} Msun')

plt.title('3) Mass Profile: M33', fontsize=15)
plt.xlabel('r (kpc)', fontsize=14)
plt.ylabel('Enclosed Mass (Msun)', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.savefig("3 Mass Profile M33.png", dpi=300, bbox_inches="tight")
plt.show()

# Define radii array for rotation curves (0.1 to 300 kpc)
r_circ = np.linspace(0.1, 300, 1000)

##################################
# 4) Milky Way Rotation Curve
##################################
Vhalo_MW = MW.CircularVelocity(1, r_circ)
Vdisk_MW = MW.CircularVelocity(2, r_circ)
Vbulge_MW = MW.CircularVelocity(3, r_circ)
Vtotal_MW = MW.CircularVelocityTotal(r_circ)

# Reuse the big-radius total halo mass from Cell 2: MW_halo_mass_total
MW_Vhern = MW.HernquistVCirc(r_circ, a_guess_MW, MW_halo_mass_total)

plt.figure(figsize=(8,6))
plt.plot(r_circ, Vhalo_MW, 'b', label='Halo')
plt.plot(r_circ, Vdisk_MW, 'r', label='Disk')
plt.plot(r_circ, Vbulge_MW, 'g', label='Bulge')
plt.plot(r_circ, Vtotal_MW, 'k', label='Total')

# Overplot the Hernquist halo velocity
plt.plot(r_circ, MW_Vhern, 'b--',
         label=f'Hernquist Halo (a={a_guess_MW}, Mhalo={MW_halo_mass_total:.2e})')

plt.title('4) Rotation Curve: Milky Way', fontsize=15)
plt.xlabel('r (kpc)', fontsize=14)
plt.ylabel('Vcirc (km/s)', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.savefig("4 Rotation Curve MilkyWay.png", dpi=300, bbox_inches="tight")
plt.show()


##################################
# 5) M31 Rotation Curve
##################################
Vhalo_M31 = M31.CircularVelocity(1, r_circ)
Vdisk_M31 = M31.CircularVelocity(2, r_circ)
Vbulge_M31 = M31.CircularVelocity(3, r_circ)
Vtotal_M31 = M31.CircularVelocityTotal(r_circ)

M31_Vhern = M31.HernquistVCirc(r_circ, a_guess_M31, M31_halo_mass_total)

plt.figure(figsize=(8,6))
plt.plot(r_circ, Vhalo_M31, 'b', label='Halo')
plt.plot(r_circ, Vdisk_M31, 'r', label='Disk')
plt.plot(r_circ, Vbulge_M31, 'g', label='Bulge')
plt.plot(r_circ, Vtotal_M31, 'k', label='Total')

plt.plot(r_circ, M31_Vhern, 'b--',
         label=f'Hernquist Halo (a={a_guess_M31}, Mhalo={M31_halo_mass_total:.2e})')

plt.title('5) Rotation Curve: M31', fontsize=15)
plt.xlabel('r (kpc)', fontsize=14)
plt.ylabel('Vcirc (km/s)', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.savefig("5 Rotation Curve M31.png", dpi=300, bbox_inches="tight")
plt.show()


##################################
# 6) M33 Rotation Curve
##################################
Vhalo_M33 = M33.CircularVelocity(1, r_circ)
Vdisk_M33 = M33.CircularVelocity(2, r_circ)
Vbulge_M33 = 0.0 * Vhalo_M33  # M33 has no bulge
Vtotal_M33 = M33.CircularVelocityTotal(r_circ)

M33_Vhern = M33.HernquistVCirc(r_circ, a_guess_M33, M33_halo_mass_total)

plt.figure(figsize=(8,6))
plt.plot(r_circ, Vhalo_M33, 'b', label='Halo')
plt.plot(r_circ, Vdisk_M33, 'r', label='Disk')
plt.plot(r_circ, Vbulge_M33, 'g', label='Bulge (none)')
plt.plot(r_circ, Vtotal_M33, 'k', label='Total')

plt.plot(r_circ, M33_Vhern, 'b--',
         label=f'Hernquist Halo (a={a_guess_M33}, Mhalo={M33_halo_mass_total:.2e})')

plt.title('6) Rotation Curve: M33', fontsize=15)
plt.xlabel('r (kpc)', fontsize=14)
plt.ylabel('Vcirc (km/s)', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.savefig("6 Rotation Curve M33.png", dpi=300, bbox_inches="tight")
plt.show()




# ===== File: HW6OrbitCOM.py =====
# Cell 1

# Homework 6 Template
# G. Besla & R. Li

#cell 2

# import modules
import numpy as np
import astropy.units as u
from astropy.constants import G

# import plotting modules
import matplotlib.pyplot as plt
import matplotlib
#%matplotlib inline

# my modules
from ReadFile import Read
# Step 1: modify CenterOfMass so that COM_P now takes a parameter specifying 
# by how much to decrease RMAX instead of a factor of 2
from CenterOfMass2 import CenterOfMass

#cell 3

def OrbitCOM(galaxy, start, end, n):
    """
    function that loops over all the desired snapshots to compute the COM pos and vel 
    as a function of time.
    
    inputs:
    -------
    galaxy : str
        'MW', 'M31', 'M33'
    start  : int
        snapshot to start from
    end    : int
        snapshot to end on
    n      : int
        step size for reading snapshots
        
    outputs:
    --------
    A file "Orbit_<galaxy>.txt" containing columns:
    t, x, y, z, vx, vy, vz
    
    We only do this once and store the data so we don't 
    keep repeating this process.
    """

    # compose the filename for output
    fileout = f"Orbit_{galaxy}.txt"
    
    # set tolerance and volDec
    # For MW, M31 we can use volDec=2; for M33, use volDec=4 (since M33 is more stripped)
    delta = 0.1
    if galaxy == "M33":
        volDec = 4.0
    else:
        volDec = 2.0

    # generate the snapshot id sequence 
    snap_ids = np.arange(start, end+1, n)  # e.g. 0, 5, 10, ... 800
    # simple check:
    if len(snap_ids) == 0:
        print("No snapshots found. Check your inputs.")
        return

    # initialize the array for orbital info: t, x, y, z, vx, vy, vz
    orbit = np.zeros((len(snap_ids), 7))

    # a for loop to loop over files
    for i, snap_id in enumerate(snap_ids):
        
        # compose the data filename 
        # if using local directories named MW, M31, M33 for the files, do:
        ilbl = f"{snap_id:03d}"   # ensures a three-digit label
        filename = f"{galaxy}/{galaxy}_{ilbl}.txt"

        # Initialize an instance of CenterOfMass class, using DISK particles
        COM = CenterOfMass(filename, 2)

        # Store the COM pos and vel. COM_P now has volDec
        com_pos = COM.COM_P(delta=delta, volDec=volDec)
        com_vel = COM.COM_V(com_pos[0], com_pos[1], com_pos[2])
        
        # store the time, pos, vel in ith element of the orbit array,  without units (.value)
        # time is in Myr in the snapshot, so dividing by 1000 to get Gyr
        orbit[i, 0] = COM.time.to(u.Gyr).value
        orbit[i, 1] = com_pos[0].value # x
        orbit[i, 2] = com_pos[1].value # y
        orbit[i, 3] = com_pos[2].value # z
        orbit[i, 4] = com_vel[0].value # vx
        orbit[i, 5] = com_vel[1].value # vy
        orbit[i, 6] = com_vel[2].value # vz
        
        # print snap_id to see the progress
        print(f"Done with snapshot {snap_id}")

    # write the data to a file
    # we do this because we don't want to have to repeat this process 
    # this code should only have to be called once per galaxy.
    np.savetxt(fileout, orbit, fmt = "%11.3f"*7, comments='#',
               header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}"\
                      .format('t', 'x', 'y', 'z', 'vx', 'vy', 'vz'))
    print(f"Orbit saved to {fileout}")
    
#Cell 4

# Recover the orbits and generate the COM files for each galaxy
# read in 800 snapshots in intervals of n=5
# Note: This might take a little while - test your code with a smaller number of snapshots first! 

OrbitCOM("MW", 0, 800, 5)
OrbitCOM("M31", 0, 800, 5)
OrbitCOM("M33", 0, 800, 5)

#cell 5

# Read in the data files for the orbits of each galaxy that you just created
# headers:  t, x, y, z, vx, vy, vz
# using np.genfromtxt

import numpy as np
import matplotlib.pyplot as plt

dataMW  = np.genfromtxt("Orbit_MW.txt", comments='#')
dataM31 = np.genfromtxt("Orbit_M31.txt", comments='#')
dataM33 = np.genfromtxt("Orbit_M33.txt", comments='#')

# columns: t, x, y, z, vx, vy, vz
t_MW  = dataMW[:, 0]
x_MW  = dataMW[:, 1]
y_MW  = dataMW[:, 2]
z_MW  = dataMW[:, 3]
vx_MW = dataMW[:, 4]
vy_MW = dataMW[:, 5]
vz_MW = dataMW[:, 6]

t_M31  = dataM31[:, 0]
x_M31  = dataM31[:, 1]
y_M31  = dataM31[:, 2]
z_M31  = dataM31[:, 3]
vx_M31 = dataM31[:, 4]
vy_M31 = dataM31[:, 5]
vz_M31 = dataM31[:, 6]

t_M33  = dataM33[:, 0]
x_M33  = dataM33[:, 1]
y_M33  = dataM33[:, 2]
z_M33  = dataM33[:, 3]
vx_M33 = dataM33[:, 4]
vy_M33 = dataM33[:, 5]
vz_M33 = dataM33[:, 6]

#cell 6 

# function to compute the magnitude of the difference between two vectors 
# You can use this function to return both the relative position and relative velocity for two 
# galaxies over the entire orbit  

# function to compute the magnitude of the difference between two vectors 
def VectorDiffMag(x1, y1, z1, x2, y2, z2):
    """
    Returns the magnitude of the difference between
    vector1(x1, y1, z1) and vector2(x2, y2, z2).
    """
    return np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2 )

def VectorDiffMagVel(vx1, vy1, vz1, vx2, vy2, vz2):
    """
    Returns the magnitude of velocity difference
    between vector1(vx1, vy1, vz1) and vector2(vx2, vy2, vz2).
    """
    return np.sqrt( (vx2 - vx1)**2 + (vy2 - vy1)**2 + (vz2 - vz1)**2 )

# We assume t_MW, t_M31, t_M33 have the same shape 
# (they should, if they were written out the same way). 
# But let's do a quick check or at least assume they match.

#Cell 7

# Determine the magnitude of the relative position and velocities 

# of MW and M31
r_MW_M31  = VectorDiffMag(x_MW,  y_MW,  z_MW,  x_M31,  y_M31,  z_M31)
v_MW_M31  = VectorDiffMagVel(vx_MW, vy_MW, vz_MW, vx_M31, vy_M31, vz_M31)

# of M33 and M31
r_M33_M31 = VectorDiffMag(x_M33, y_M33, z_M33, x_M31, y_M31, z_M31)
v_M33_M31 = VectorDiffMagVel(vx_M33, vy_M33, vz_M33, vx_M31, vy_M31, vz_M31)

#cell 8

# Plot the Orbit of the galaxies 
#################################
# Plot the orbital velocities of the galaxies 
#################################

fig, ax = plt.subplots(1,2, figsize=(25,10))

# Left subplot: separation vs time
ax[0].plot(t_MW, r_MW_M31, color='blue',  label='MW-M31')
ax[0].plot(t_M31, r_M33_M31, color='red', label='M33-M31')
ax[0].set_xlabel('Time (Gyr)')
ax[0].set_ylabel('Separation (kpc)')
ax[0].set_title('Separation vs Time')
ax[0].legend()

# Right subplot: relative speed vs time
ax[1].plot(t_MW, v_MW_M31, color='blue',  label='MW-M31')
ax[1].plot(t_M31, v_M33_M31, color='red', label='M33-M31')
ax[1].set_xlabel('Time (Gyr)')
ax[1].set_ylabel('Relative Speed (km/s)')
ax[1].set_title('Relative Velocity vs Time')
ax[1].legend()

plt.tight_layout()
plt.show()

# Question 4A : Identify local minima in r_MW_M31 and count only MAJOR close encounters

def find_local_minima(x):
    """
    Returns a list of indices i where x[i] is a local minimum:
    x[i] < x[i-1] and x[i] < x[i+1].
    """
    minima_indices = []
    for i in range(1, len(x)-1):
        if (x[i] < x[i-1]) and (x[i] < x[i+1]):
            minima_indices.append(i)
    return minima_indices

def find_local_maxima(x):
    """
    Returns a list of indices i where x[i] is a local maximum:
    x[i] > x[i-1] and x[i] > x[i+1].
    """
    maxima_indices = []
    for i in range(1, len(x)-1):
        if (x[i] > x[i-1]) and (x[i] > x[i+1]):
            maxima_indices.append(i)
    return maxima_indices

# Find all local minima in separation
r_min_indices = find_local_minima(r_MW_M31)
r_max_indices = find_local_maxima(r_MW_M31)  # To find previous peak before dip

# Define lower and upper bounds for close encounters
lower_thresh = 1     # Encounters must be closer than this
upper_thresh = 50    # Ignore separations above this
major_dip = 1       # Must be at least 1 kpc lower than previous peak

filtered_encounters = []

# Loop through minima and check criteria
for idx in r_min_indices:
    if lower_thresh <= r_MW_M31[idx] <= upper_thresh:  # Within range
        # Find the most recent maximum before this minimum
        prev_max_idx = max([m for m in r_max_indices if m < idx], default=None)

        if prev_max_idx is not None:
            drop = r_MW_M31[prev_max_idx] - r_MW_M31[idx]  # How much separation dropped
            if drop >= major_dip:  # Only count if drop is significant
                filtered_encounters.append(idx)

# Print the results
print(f"\nMajor Close Encounters (1 < r < 50 kpc & Δr >= {major_dip} kpc):")
for i, idx in enumerate(filtered_encounters):
    print(f"{i+1}) t={t_MW[idx]:.2f} Gyr, r={r_MW_M31[idx]:.2f} kpc")

print(f"\nTotal number of major encounters = {len(filtered_encounters)}")

# '''
# Question 4A
#
# visually its 3 close encounters before it then goes off to zero eventually with changes that this code shows 
# and the output reveals more encounters.
#
# the issue that you see with my code is while it is able to pick up absolute minute details in the shifts of separation 
# because of the ability to adjust threshold, you are not yet able to tell the code that you want it to select only 
# major points of cross over which is why it reveals 9 points.
# 
# I am working on updating this so that you can set a threshhold upper and lower limit to try and see what options 
# are available.
#
# Update - tried a new technique for the thresholding that has worked and now the code also shows 3 close encounters.
# 
# '''
# Question 4B : Relationship between separation & relative velocity

# ''' 
# Question 4B
# 
# they are inversely proportianate to each other as you can see that in the relation of MW and M31, as the separation comes to a sudden decrease you see a spike in the velocity graph skyrocket.
# then as the separation gets further you see that the velocity slows down again, and then repeats that process till they eventually come to a natural zero point where they are within close enough
# distance to each other that you then see that they separation and the velocity are approaching 0, which in this case is still incerdibly fast.
# 
# this is different for the relation of M33 and M31, where as you can see the shifting of the 2 galaxies coming together also causes them to go out far enough as well. however what you are able to
# see again is the inverse proportion to the lines themselves which goes to further prove the point discussed above.
# 
# '''
# Question 4C : Checking final separation + zoomed/log plot for MW-M31

import numpy as np
import matplotlib.pyplot as plt

# 1) Check final separation
final_sep = r_MW_M31[-1]
final_time = t_MW[-1]
print(f"Final snapshot time ~ {final_time:.2f} Gyr, separation = {final_sep:.2f} kpc")

# If you want to see if/when r < 10 kpc at any time:
merge_idx = np.where(r_MW_M31 < 10)[0]
if len(merge_idx) > 0:
    print("Times when MW-M31 separation < 10 kpc:")
    for i in merge_idx:
        print(f"  t={t_MW[i]:.3f} Gyr, r={r_MW_M31[i]:.3f} kpc")
else:
    print("No times found with MW-M31 separation < 10 kpc in our snapshots.")

# 2) Zoomed plot: we can focus on t>4 Gyr or t>5 Gyr for a better look
zoom_mask = np.where(t_MW > 4.0)[0]

fig, ax = plt.subplots(1,2, figsize=(14,5))

# Left: normal scale
ax[0].plot(t_MW[zoom_mask], r_MW_M31[zoom_mask], label='MW-M31')
ax[0].plot(t_M31[zoom_mask], r_M33_M31[zoom_mask], label='M33-M31')
ax[0].set_xlabel('Time (Gyr)')
ax[0].set_ylabel('Separation (kpc)')
ax[0].set_title('Zoomed: t > 4 Gyr')
ax[0].legend()

# Right: log scale on y
ax[1].semilogy(t_MW[zoom_mask], r_MW_M31[zoom_mask], label='MW-M31')
ax[1].semilogy(t_M31[zoom_mask], r_M33_M31[zoom_mask], label='M33-M31')
ax[1].set_xlabel('Time (Gyr)')
ax[1].set_ylabel('Separation (kpc) [log scale]')
ax[1].set_title('Log scale: t > 4 Gyr')
ax[1].legend()

plt.tight_layout()
plt.show()

# 3) Checking M33’s orbit after that merge time:
# e.g. if the last close approach is near t~5 Gyr, let's see r_M33_M31 after that
merge_time = 5.0  # example
m33_indices_after_merge = np.where(t_M33 >= merge_time)[0]
print(f"Average M33-M31 separation after t={merge_time} Gyr: "
      f"{np.mean(r_M33_M31[m33_indices_after_merge]):.2f} kpc")


# '''
# Question 4C
# 
# Final snapshot time ~ 11.43 Gyr, separation = ~1.43 kpc
# Times when MW-M31 separation < 10 kpc: (long list)
# Average M33-M31 separation after t=5.0 Gyr: ~83.21 kpc
# 
# The figure also shows that around 5-6 Gyr we see them bouncing 
# below 10 kpc multiple times, effectively merging by ~6-8 Gyr 
# and staying near 1-2 kpc from ~10 Gyr onward. M33 remains 
# tens of kpc away, with repeated orbits.
# 
# '''
# Question 4D : M33 orbit decay rate after 6 Gyr

# We'll find apocenters for M33-M31 after t=6 Gyr. Apocenters = local maxima
def find_local_maxima(x):
    """Return indices i where x[i] is a local maximum."""
    maxima_indices = []
    for i in range(1, len(x)-1):
        if (x[i] > x[i-1]) and (x[i] > x[i+1]):
            maxima_indices.append(i)
    return maxima_indices

t_cut = 6.0
idx_6Gyr = np.where(t_M33 >= t_cut)[0]
apocenter_indices = []

# We'll only search local maxima in the portion t>=6 Gyr
# so we limit ourselves to the range [idx_6Gyr[0], len(r_M33_M31)-2]
start_i = idx_6Gyr[0]
end_i   = len(r_M33_M31) - 1
for i in range(start_i+1, end_i):
    if (r_M33_M31[i] > r_M33_M31[i-1]) and (r_M33_M31[i] > r_M33_M31[i+1]):
        apocenter_indices.append(i)

print("M33 Apocenters after 6 Gyr:")
for i in apocenter_indices:
    print(f"  t={t_M33[i]:.2f} Gyr, r={r_M33_M31[i]:.2f} kpc")

# If we have at least 2 apocenters, measure the difference
if len(apocenter_indices) >= 2:
    i1 = apocenter_indices[0]
    i2 = apocenter_indices[1]
    rA1 = r_M33_M31[i1]
    rA2 = r_M33_M31[i2]
    tA1 = t_M33[i1]
    tA2 = t_M33[i2]

    dr = rA1 - rA2
    dt = tA2 - tA1
    decay_rate = dr/dt  # kpc/Gyr
    print(f"\nApprox. decay rate between first two apocenters: {decay_rate:.2f} kpc/Gyr")

    # Estimate how long M33 would take to merge if it starts ~75 kpc away
    # time = distance / rate
    # But if rate is negative => r is decreasing
    future_dist = 75.0
    time_to_merge = future_dist / decay_rate if decay_rate>0 else None
    if time_to_merge and time_to_merge>0:
        print(f"Estimated time for M33 to merge from 75 kpc: {time_to_merge:.2f} Gyr")
    else:
        print("Decay rate is negative or zero -- can't estimate a straightforward time to merge.")
else:
    print("Not enough apocenters after 6 Gyr to estimate a decay rate.")

# '''
# Question 4D
# 
# M33 Apocenters after 6 Gyr:
#   t=7.50 Gyr, r=108.76 kpc
#   t=8.93 Gyr, r=89.08 kpc
#   t=10.07 Gyr, r=77.42 kpc
#   t=11.07 Gyr, r=70.71 kpc
# 
# Approx. decay rate between first two apocenters: 13.77 kpc/Gyr
# Estimated time for M33 to merge from 75 kpc: 5.45 Gyr
# 
# '''



# ===== File: HW7Garg.py =====
 # Make edits where instructed - look for "****", which indicates where you need to add code. 
# ---------------------------- Cell 1 ---------------------------- #

# Homework 7 Template
# Rixin Li & G. Besla
#
# Make edits where instructed - look for "****", which indicates where you need to 
# add code. 

# import necessary modules
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as const

# import Latex module so we can display the results with symbols
from IPython.display import Latex

# **** import CenterOfMass to determine the COM pos/vel of M33
from CenterOfMass import CenterOfMass

# **** import the GalaxyMass to determine the mass of M31 for each component
from GalaxyMass import ComponentMass

# Cell 2

class M33AnalyticOrbit:
    """ Calculate the analytical orbit of M33 around M31 """
    
    def __init__(self, outfilename):
        """ **** ADD COMMENTS 
        
        Initialize the M33AnalyticOrbit class, setting up all relevant 
        parameters for M33's orbit about M31 using their positions and 
        velocities at snapshot 0.
        
        Inputs:
        -------
        outfilename : str
            The name of the output file where we'll save the integrated orbit.
        """
        
        ### get the gravitational constant (the value is 4.498768e-06 in kpc^3/Msun/Gyr^2)
        self.G = 4.498768e-6
        
        ### **** store the output file name
        self.filename = outfilename
        
        ### get the current pos/vel of M33 
        # **** create an instance of the  CenterOfMass class for M33 
        COM_M33 = CenterOfMass("M33_000.txt", 2)  # 2=disk
        posM33 = COM_M33.COM_P(delta=0.1) 
        velM33 = COM_M33.COM_V(posM33[0], posM33[1], posM33[2])
        
        # **** store the position VECTOR of the M33 COM (.value to get rid of units)
        posM33 = posM33.value
        velM33 = velM33.value
        
        ### get the current pos/vel of M31 
        COM_M31 = CenterOfMass("M31_000.txt", 2)
        posM31 = COM_M31.COM_P(delta=0.1)
        velM31 = COM_M31.COM_V(posM31[0], posM31[1], posM31[2])
        
        posM31 = posM31.value
        velM31 = velM31.value
        
        ### store the DIFFERENCE between the vectors posM33 - posM31
        # **** create two VECTORs self.r0 and self.v0 and have them be the
        # relative position and velocity VECTORS of M33
        self.r0 = posM33 - posM31
        self.v0 = velM33 - velM31
        
        
        ### get the mass of each component in M31 
        ### disk
        # **** self.rdisk = scale length (no units)
        self.rdisk = 5.0
        
        # **** self.Mdisk set with ComponentMass function. 
        #      * 1e12 to convert from 1e12 Msun to Msun 
        self.Mdisk  = 1e12*ComponentMass("M31_000.txt", 2)
        
        ### bulge
        # **** self.rbulge = set scale length (no units)
        self.rbulge = 1.0
        
        # **** self.Mbulge  set with ComponentMass function
        self.Mbulge = 1e12*ComponentMass("M31_000.txt", 3)
        
        ### Halo
        # **** self.rhalo = set scale length from HW5 (no units)
        # (the instructions say to use 60.0)
        self.rhalo = 60.0
        
        # **** self.Mhalo set with ComponentMass function
        self.Mhalo = 1e12*ComponentMass("M31_000.txt", 1)
    
    
    
    def HernquistAccel(self, M, ra, r):
        """ **** ADD COMMENTS 
        
        Computes the gravitational acceleration from a Hernquist profile:
        
            a = -G M / [r_mag (r_a + r_mag)^2] * r_vec
        
        Inputs:
        -------
        M   : float
            Total mass of the halo or bulge (Msun)
        ra  : float
            Scale radius (kpc)
        r   : np.ndarray of shape (3,)
            Position vector [x, y, z] in kpc
        
        Returns:
        --------
        accel : np.ndarray of shape (3,)
            Acceleration vector in kpc/Gyr^2
        """
        
        # store the magnitude of the position vector
        rmag = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
        
        # avoid divide-by-zero issues if rmag=0
        if rmag == 0:
            return np.array([0.0, 0.0, 0.0])
        
        # store the acceleration vector 
        # a = - G * M / [rmag * (ra + rmag)^2] * r
        Hern = - self.G * M / (rmag*(ra + rmag)**2) * r
        
        return Hern
    
    
    
    def MiyamotoNagaiAccel(self, M, rd, r):
        """ **** ADD COMMENTS 

        Computes the gravitational acceleration from a Miyamoto-Nagai disk profile:
        
        a = -G M / (R^2 + B^2)^(3/2) * r * [1, 1, B / sqrt(z^2 + z_d^2] 
        where 
            R = sqrt(x^2 + y^2)
            B = rd + sqrt(z^2 + z_d^2)
            z_d = rd/5.0
        """
        
        x, y, z = r
        # define R = sqrt(x^2 + y^2)
        R = np.sqrt(x**2 + y**2)
        
        # define z_d:
        z_d = rd/5.0
        # define B
        B = rd + np.sqrt(z**2 + z_d**2)
        
        # define (R^2 + B^2)^(3/2)
        denom = (R**2 + B**2)**1.5
        
        # again handle the case if we happen to be at r=0:
        if denom == 0:
            return np.array([0.0, 0.0, 0.0])
        
        # The standard formula is:
        # ax = -G M x / (R^2 + B^2)^(3/2)
        # ay = -G M y / (R^2 + B^2)^(3/2)
        # az = -G M B z / [ (R^2 + B^2)^(3/2) sqrt(z^2 + z_d^2) ]
        
        # We can do this in a vector form as in eqn (4) in the instructions:
        # a = -GM / (R^2 + B^2)^(3/2) * r * [1, 1, B/sqrt(z^2 + z_d^2)]
        
        # define the factor -G M / (R^2 + B^2)^(3/2)
        factor = - self.G * M / denom
        
        # the z correction factor:
        zfactor = B / np.sqrt(z**2 + z_d**2)
        
        # build a small array to multiply with r:
        multi = np.array([1.0, 1.0, zfactor])
        
        # the final acceleration vector:
        a_disk = factor * r * multi
        
        return a_disk
    
    
    def M31Accel(self, r):
        """ **** ADD COMMENTS 
        
        Computes the total acceleration from M31's halo, bulge, and disk 
        at the position vector r.
        
        Inputs:
        -------
        r : np.ndarray of shape (3,)
            Relative position vector in kpc

        Returns:
        --------
        a_total : np.ndarray of shape (3,)
            Sum of all acceleration components from M31: halo+bulge+disk
        """
        
        # Call the HernquistAccel for the halo
        a_halo  = self.HernquistAccel(self.Mhalo,  self.rhalo,  r)
        
        # Call the HernquistAccel for the bulge
        a_bulge = self.HernquistAccel(self.Mbulge, self.rbulge, r)
        
        # Call the MiyamotoNagaiAccel for the disk
        a_disk  = self.MiyamotoNagaiAccel(self.Mdisk, self.rdisk, r)
        
        # Sum them up (vector sum)
        a_total = a_halo + a_bulge + a_disk
        
        return a_total
    
    
    def LeapFrog(self, r, v, dt):
        """
        Advance the position and velocity of M33 by one timestep dt 
        using the Leap Frog integrator.
        
        Parameters
        ----------
        r : np.ndarray
            Current position vector [x, y, z] of M33 relative to M31
        v : np.ndarray
            Current velocity vector [vx, vy, vz] of M33 relative to M31
        dt : float
            Timestep in Gyr; can be positive or negative
        
        Returns
        -------
        rnew : np.ndarray
            Updated position vector after one timestep
        vnew : np.ndarray
            Updated velocity vector after one timestep
        
        The integration scheme:
        
        1) Predict the half-step position:
            r_half = r + v*(dt/2)
            
        2) Compute acceleration a_half at r_half:
            a_half = M31Accel(r_half)
            
        3) Update the velocity at the full step:
            v_new = v + a_half*dt
            
        4) Update the position at the full step 
           (using r_half and v_new):
            r_new = r_half + v_new*(dt/2)
        """
        
        # 1) Predict the half-step position
        rhalf = r + v*(dt/2.0)
        
        # 2) Compute the acceleration at the half-step
        a_half = self.M31Accel(rhalf)
        
        # 3) Update velocity to the next full step
        vnew = v + a_half*dt
        
        # 4) Update position to the next full step
        rnew = rhalf + vnew*(dt/2.0)
        
        return rnew, vnew
    
    def OrbitIntegration(self, t0, dt, tmax):
        """
        Integrate M33's orbit forward in time from t0 to tmax using 
        the LeapFrog method defined in LeapFrog().
        
        Parameters
        ----------
        t0   : float
            Initial time in Gyr (e.g., 0.0)
        dt   : float
            Timestep in Gyr (e.g., 0.01)
        tmax : float
            Final time in Gyr (e.g., 10.0)
        
        Returns
        -------
        None. The orbit array is written to self.filename as a txt file
        with columns: [t, x, y, z, vx, vy, vz].
        """
        
        # initialize the time
        t = t0
        
        # estimate how many steps we need
        # e.g. if dt=0.01 Gyr and tmax=10 Gyr, we might have 1000 steps
        # but tmax can be anything, so let's use int(...) plus some buffer
        nsteps = int( (tmax - t0)/dt ) + 2
        
        # initialize an array to store time, x, y, z, vx, vy, vz
        # shape: (nsteps, 7)
        orbit = np.zeros( (nsteps, 7) )
        
        # current position and velocity from the initialization
        rcurrent = self.r0
        vcurrent = self.v0
        
        # store the initial conditions in the first row
        # orbit[0] = t0, rx, ry, rz, vx, vy, vz
        orbit[0,0] = t
        orbit[0,1:4] = rcurrent
        orbit[0,4:7] = vcurrent
        
        # index counter for array
        i = 1
        
        # start the integration loop
        while (t < tmax) and (i < nsteps):
            
            # advance time by dt
            t += dt
            
            # use LeapFrog to get new position and velocity
            rnew, vnew = self.LeapFrog(rcurrent, vcurrent, dt)
            
            # store the new time and phase-space coordinates
            orbit[i,0]   = t
            orbit[i,1:4] = rnew
            orbit[i,4:7] = vnew
            
            # update rcurrent, vcurrent for next iteration
            rcurrent = rnew
            vcurrent = vnew
            
            # increment index
            i += 1
        
        # now that we've finished, let's trim the orbit array 
        # in case we didn't use all rows:
        orbit = orbit[:i]
        
        # write the array to file
        # e.g. columns: t, x, y, z, vx, vy, vz
        np.savetxt(self.filename, orbit, fmt="%11.3f", comments='#',
                   header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}"
                   .format('t','x','y','z','vx','vy','vz'))
        
# Return the final orbit array
M33orbit = M33AnalyticOrbit("M33_analytic.txt")
M33orbit.OrbitIntegration(t0=0.0, dt=0.01, tmax=10.0)
print('File created')

# ----------------- Code for Part 5 question 1 ----------------- #

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1) Read ANALYTIC orbit
# ------------------------------
data_analytic = np.genfromtxt("M33_analytic.txt", comments='#')
# columns: t, x, y, z, vx, vy, vz
tA  = data_analytic[:,0]
xA  = data_analytic[:,1]
yA  = data_analytic[:,2]
zA  = data_analytic[:,3]
vxA = data_analytic[:,4]
vyA = data_analytic[:,5]
vzA = data_analytic[:,6]

rA = np.sqrt(xA**2 + yA**2 + zA**2)    # total distance
vA = np.sqrt(vxA**2 + vyA**2 + vzA**2) # total speed

# ------------------------------
# 2) Read SIMULATED orbits from HW6
#    We want M33 relative to M31
# ------------------------------
dataM31 = np.genfromtxt("Orbit_M31.txt", comments='#')
dataM33 = np.genfromtxt("Orbit_M33.txt", comments='#')
# columns: t, x, y, z, vx, vy, vz

t_M31  = dataM31[:,0]
x_M31  = dataM31[:,1]
y_M31  = dataM31[:,2]
z_M31  = dataM31[:,3]
vx_M31 = dataM31[:,4]
vy_M31 = dataM31[:,5]
vz_M31 = dataM31[:,6]

t_M33  = dataM33[:,0]
x_M33  = dataM33[:,1]
y_M33  = dataM33[:,2]
z_M33  = dataM33[:,3]
vx_M33 = dataM33[:,4]
vy_M33 = dataM33[:,5]
vz_M33 = dataM33[:,6]

x_rel = x_M33 - x_M31
y_rel = y_M33 - y_M31
z_rel = z_M33 - z_M31

vx_rel = vx_M33 - vx_M31
vy_rel = vy_M33 - vy_M31
vz_rel = vz_M33 - vz_M31

rSim = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)    
vSim = np.sqrt(vx_rel**2 + vy_rel**2 + vz_rel**2)

tSim = t_M33  # same as t_M31

# ------------------------------
# 3) Plot the results
# ------------------------------
fig, axes = plt.subplots(1,2, figsize=(16,6))

# Left: distance vs time
axes[0].plot(tA,  rA,  label='M33-M31 predicted',  color='blue')
axes[0].plot(tSim, rSim, label='M33-M31 simulation', color='red')
axes[0].set_xlabel('Time (Gyr)', fontsize=14)
axes[0].set_ylabel('Distance (kpc)', fontsize=14)
axes[0].set_title('M33 Distance from M31', fontsize=15)
axes[0].legend()

# Right: speed vs time
axes[1].plot(tA,  vA,  label='M33-M31 predicted', color='blue')
axes[1].plot(tSim, vSim, label='M33-M31 simulation', color='red')
axes[1].set_xlabel('Time (Gyr)', fontsize=14)
axes[1].set_ylabel('Speed (km/s)', fontsize=14)
axes[1].set_title('M33 Velocity relative to M31', fontsize=15)
axes[1].legend()

plt.tight_layout()
plt.show()

# analysis for Part 5 question 2 - this was just for a visual representation nothing else #

import numpy as np
import matplotlib.pyplot as plt

# 1) Read in or define the arrays from your existing code

rA_interp = np.interp(tSim, tA, rA)
vA_interp = np.interp(tSim, tA, vA)

# 2) Compute the difference or ratio
r_diff  = rSim - rA_interp
v_diff  = vSim - vA_interp

# or relative difference
# e.g. ratio = (Sim - Ana) / Ana
r_ratio = r_diff / rA_interp
v_ratio = v_diff / vA_interp

# 3) Plot the differences vs time
fig, ax = plt.subplots(1, 2, figsize=(14,5))

# Left: Distance difference
ax[0].plot(tSim, r_diff, 'b', label='Distance difference (kpc)')
ax[0].axhline(0, color='k', ls='--', lw=1)
ax[0].set_xlabel('Time (Gyr)', fontsize=14)
ax[0].set_ylabel('r_Sim - r_Ana (kpc)', fontsize=14)
ax[0].legend()

# Right: Velocity difference
ax[1].plot(tSim, v_diff, 'r', label='Velocity difference (km/s)')
ax[1].axhline(0, color='k', ls='--', lw=1)
ax[1].set_xlabel('Time (Gyr)', fontsize=14)
ax[1].set_ylabel('v_Sim - v_Ana (km/s)', fontsize=14)
ax[1].legend()

plt.tight_layout()
plt.show()

# 4) Print some summary statistics
print("SUMMARY STATISTICS (Sim - Analytic):")
print(f"Distance difference: mean={np.mean(r_diff):.3f} kpc,  std={np.std(r_diff):.3f} kpc")
print(f"Velocity difference: mean={np.mean(v_diff):.3f} km/s, std={np.std(v_diff):.3f} km/s")

# You could also check max difference:
print(f"Max distance difference = {np.max(np.abs(r_diff)):.3f} kpc at t={tSim[np.argmax(np.abs(r_diff))]:.2f} Gyr")
print(f"Max velocity difference = {np.max(np.abs(v_diff)):.3f} km/s at t={tSim[np.argmax(np.abs(v_diff))]:.2f} Gyr")




