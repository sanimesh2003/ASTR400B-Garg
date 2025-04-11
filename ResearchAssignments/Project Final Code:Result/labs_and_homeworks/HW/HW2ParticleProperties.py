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


