import numpy as np  
from ReadFile import Read  

def ComponentMass(filename, particle_type):
        time, total_particles, data = Read(filename)

        index = np.where(data['type'] == particle_type)
        mass = np.sum(data['m'][index]) 

        mass = mass * 1e-2  

        return np.round(mass, 3)

if __name__ == "__main__":
    filename = "MW_000.txt"  
    halo_mass = ComponentMass(filename, 1) 
    disk_mass = ComponentMass(filename, 2) 
    bulge_mass = ComponentMass(filename, 3)  

