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


