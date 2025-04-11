# PROJECT CODE V4 - Part 1

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import os
import pandas as pd
import glob
from scipy.optimize import curve_fit

# GLOBAL PLOTTING STYLE
plt.rcParams["figure.figsize"] = (8,6)
plt.rcParams["font.size"]      = 14
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["legend.fontsize"] = 12

print("SECTION 1 COMPLETE: Necessary imports and global setup done.")

# PROJECT CODE V4 - Part 2
# PURPOSE:
#   1) Define a function "Read" that reads our snapshot files from scratch
#   2) Demonstrate reading snapshot data for MW, M31, M33
#   3) (Optionally) store or display relevant info
#
# We'll re-implement the logic that was in 1ReadFile.py (or F1ReadFile.py),
# but from scratch, directly in this notebook, to avoid external imports.

def Read(filename):
    """
    Function to read a GADGET-style snapshot data file and return:
      1) Time in Myr
      2) Total number of particles
      3) A structured numpy array with fields: [type, m, x, y, z, vx, vy, vz]
    PARAMETERS
    ----------
    filename : str
        Path to the data file, e.g. 'MW_000.txt'
    RETURNS
    -------
    time  : float
        The snapshot time in Myr
    total : int
        The total number of particles in the file
    data  : np.ndarray
        A structured array containing columns:
          'type', 'm', 'x', 'y', 'z', 'vx', 'vy', 'vz'
    """

    file = open(filename, 'r')

    line1 = file.readline()
    # We'll parse out the time from this line
    # We'll split and take the last item as the time in Myr
    label, value = line1.split()
    time = float(value)  # Myr

    line2 = file.readline()
    label2, value2 = line2.split()
    total = int(value2)  # total number of particles

    file.readline()
    file.readline()

    # We'll store the values in lists, then convert to structured array
    ptype_list = []
    m_list = []
    x_list = []
    y_list = []
    z_list = []
    vx_list = []
    vy_list = []
    vz_list = []

    # We'll read line-by-line for "total" lines
    for _ in range(total):
        line = file.readline()
        if not line:
            # in case of truncated file
            break

        # Each line has 8 columns: type, m, x, y, z, vx, vy, vz
        vals = line.split()
        ptype_list.append(float(vals[0]))
        m_list.append(float(vals[1]))
        x_list.append(float(vals[2]))
        y_list.append(float(vals[3]))
        z_list.append(float(vals[4]))
        vx_list.append(float(vals[5]))
        vy_list.append(float(vals[6]))
        vz_list.append(float(vals[7]))

    file.close()

    # Convert these lists to a structured array
    # We'll define a dtype that matches the columns
    dt = np.dtype([
        ('type', float),
        ('m', float),
        ('x', float),
        ('y', float),
        ('z', float),
        ('vx', float),
        ('vy', float),
        ('vz', float)
    ])

    data_array = np.zeros(total, dtype=dt)

    data_array['type'] = ptype_list
    data_array['m']    = m_list
    data_array['x']    = x_list
    data_array['y']    = y_list
    data_array['z']    = z_list
    data_array['vx']   = vx_list
    data_array['vy']   = vy_list
    data_array['vz']   = vz_list

    return time, total, data_array


print("Defined the Read() function to parse snapshot data.")

# DEMO: read one snapshot per galaxy at t=0
mw_file  = "MW_000.txt"
m31_file = "M31_000.txt"
m33_file = "M33_000.txt"

# We'll read each one
print("\nReading Milky Way Snapshot file:", mw_file)
mw_time, mw_total, mw_data = Read(mw_file)
print(f"  MW time = {mw_time} Myr, total particles = {mw_total}")

print("\nReading M31 Snapshot file:", m31_file)
m31_time, m31_total, m31_data = Read(m31_file)
print(f"  M31 time = {m31_time} Myr, total particles = {m31_total}")

print("\nReading M33 Snapshot file:", m33_file)
m33_time, m33_total, m33_data = Read(m33_file)
print(f"  M33 time = {m33_time} Myr, total particles = {m33_total}")

print("\nSECTION 2 COMPLETE: We have snapshot data for MW, M31, and M33.")

# PROJECT CODE V4 - Part 3:
# Center-of-Mass and Orbit Computations
#
# We'll define a new CenterOfMass class from scratch that:
# 1) Takes in the snapshot data array (type, m, x, y, z, vx, vy, vz)
# 2) Filters on a chosen particle type (1=halo, 2=disk, 3=bulge)
# 3) Implements the iterative COM approach for position (COM_P)
# 4) Implements a method for COM velocity (COM_V)
# 
# Then we'll compute the COM for MW, M31, and M33 as a demonstration.

class CenterOfMass:
    """
    Class to define the center of mass (COM) for a given galaxy + particle type,
    using an iterative method to refine the COM position, then a separate
    approach to compute the velocity of that COM.
    """
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

        # Filter to keep only the rows that match this ptype
        idx = np.where(data['type'] == ptype)
        
        # Store the relevant mass, positions, and velocities as arrays
        self.m  = data['m'][idx]     # array of masses
        self.x  = data['x'][idx]     # array of x positions
        self.y  = data['y'][idx]     # array of y positions
        self.z  = data['z'][idx]     # array of z positions
        self.vx = data['vx'][idx]    # array of x velocities
        self.vy = data['vy'][idx]    # array of y velocities
        self.vz = data['vz'][idx]    # array of z velocities

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

print("A new 'CenterOfMass' class has been defined from scratch.")

# 1) MW center of mass, using disk particles (ptype=2):
MW_COM_disk = CenterOfMass(mw_data, ptype=2)
MW_xcom, MW_ycom, MW_zcom = MW_COM_disk.COM_P(delta=0.1)
MW_vxcom, MW_vycom, MW_vzcom = MW_COM_disk.COM_V(MW_xcom, MW_ycom, MW_zcom)

print("\n--- Milky Way COM (Disk) ---")
print(f"Position: x={MW_xcom:.3f}, y={MW_ycom:.3f}, z={MW_zcom:.3f} (kpc)")
print(f"Velocity: vx={MW_vxcom:.3f}, vy={MW_vycom:.3f}, vz={MW_vzcom:.3f} (km/s)")


# 2) M31 center of mass, disk as well:
M31_COM_disk = CenterOfMass(m31_data, ptype=2)
M31_xcom, M31_ycom, M31_zcom = M31_COM_disk.COM_P(delta=0.1)
M31_vxcom, M31_vycom, M31_vzcom = M31_COM_disk.COM_V(M31_xcom, M31_ycom, M31_zcom)

print("\n--- M31 COM (Disk) ---")
print(f"Position: x={M31_xcom:.3f}, y={M31_ycom:.3f}, z={M31_zcom:.3f} (kpc)")
print(f"Velocity: vx={M31_vxcom:.3f}, vy={M31_vycom:.3f}, vz={M31_vzcom:.3f} (km/s)")


# 3) M33 center of mass, disk again (ptype=2):
M33_COM_disk = CenterOfMass(m33_data, ptype=2)
M33_xcom, M33_ycom, M33_zcom = M33_COM_disk.COM_P(delta=0.1)
M33_vxcom, M33_vycom, M33_vzcom = M33_COM_disk.COM_V(M33_xcom, M33_ycom, M33_zcom)

print("\n--- M33 COM (Disk) ---")
print(f"Position: x={M33_xcom:.3f}, y={M33_ycom:.3f}, z={M33_zcom:.3f} (kpc)")
print(f"Velocity: vx={M33_vxcom:.3f}, vy={M33_vycom:.3f}, vz={M33_vzcom:.3f} (km/s)")

# RELATIVE POSITIONS (e.g., M33 w.r.t M31)
dx_31_33 = M33_xcom - M31_xcom
dy_31_33 = M33_ycom - M31_ycom
dz_31_33 = M33_zcom - M31_zcom
r_31_33 = np.sqrt(dx_31_33**2 + dy_31_33**2 + dz_31_33**2)

print(f"\nM33 - M31 Separation: {r_31_33:.3f} kpc")

print("\nSECTION 3 COMPLETE: We have computed COM for each galaxy.")

galaxy_folders = ["MW", "M31", "M33"]

records = []

ptype_list = [1, 2, 3]  # 1=Halo, 2=Disk, 3=Bulge

for gal in galaxy_folders:
    pattern = f"{gal}/{gal}_*.txt"
    file_list = sorted(glob.glob(pattern))
    
    # Loop over each snapshot file in this galaxy's folder
    for filename in file_list:
        # 1) Read the data
        time_myr, total_p, data_array = Read(filename)
        
        # parse out the snapshot number from the file name, if desired
        snap_str = filename.split("_")[-1].replace(".txt","")
        
        # 2) For each ptype:
        for ptype in ptype_list:
            # Build a center-of-mass object
            com_obj = CenterOfMass(data_array, ptype=ptype)
            
            # Attempt COM position
            try:
                xcom, ycom, zcom = com_obj.COM_P(delta=0.1)
            except:
                # If for some reason it fails, set 0
                xcom, ycom, zcom = 0.0, 0.0, 0.0
            
            # Attempt velocity
            try:
                vxcom, vycom, vzcom = com_obj.COM_V(xcom, ycom, zcom, rvmax=15.0)
            except:
                # If for some reason it fails, set 0
                vxcom, vycom, vzcom = 0.0, 0.0, 0.0
            
            # Convert possible NaNs to zero
            if np.isnan(xcom) or np.isinf(xcom): xcom=0.0
            if np.isnan(ycom) or np.isinf(ycom): ycom=0.0
            if np.isnan(zcom) or np.isinf(zcom): zcom=0.0
            if np.isnan(vxcom) or np.isinf(vxcom): vxcom=0.0
            if np.isnan(vycom) or np.isinf(vycom): vycom=0.0
            if np.isnan(vzcom) or np.isinf(vzcom): vzcom=0.0
            
            # 3) Append a record to our list
            #   We'll store time, snap, galaxy name, ptype, plus COM coords.
            record = {
                "galaxy" : gal,
                "snapshot" : snap_str,
                "time_Myr" : time_myr,
                "ptype" : ptype,
                "xcom" : xcom,
                "ycom" : ycom,
                "zcom" : zcom,
                "vxcom": vxcom,
                "vycom": vycom,
                "vzcom": vzcom
            }
            records.append(record)

# Now that we've looped over everything, convert records -> DataFrame
df = pd.DataFrame(records)

# Finally, let's save to a CSV
output_filename = "All_COM.csv"
df.to_csv(output_filename, index=False)
print(f"Saved COM data to {output_filename} with {len(df)} rows.")

print("\nDone! You can now load 'All_COM.csv' later in your code for analysis.")

# PROJECT CODE V4 - Part 4a
# Define a MassProfile class
#
# PURPOSE:
# For a galaxy's snapshot data (mw_data, m31_data, or m33_data) plus that galaxy's center (xCOM,yCOM,zCOM),
# we can compute the enclosed mass at any given radius for each ptype or the total.

import numpy as np

class MassProfile:
    """
    Class that computes the mass profile for a snapshot of a galaxy,
    given the galaxy's COM position, so we can measure distances from COM.
    """
    def __init__(self, data, xCOM, yCOM, zCOM):
        """
        PARAMETERS
        ----------
        data : np.ndarray
            Structured array with columns 'type', 'm', 'x', 'y', 'z', ...
        xCOM, yCOM, zCOM : float
            The center-of-mass (in kpc) for this galaxy
        """
        self.data = data  # entire structured array
        self.xCOM = xCOM
        self.yCOM = yCOM
        self.zCOM = zCOM

        # G in convenient units: kpc (km/s)^2 / Msun
        self.G = 4.498768e-6  # typical approximate factor

    def _distance_from_com(self, x, y, z):
        """
        Internal helper function to compute 3D distance from COM
        for each particle. Returns array of distances in kpc.
        """
        dx = x - self.xCOM
        dy = y - self.yCOM
        dz = z - self.zCOM
        rr = np.sqrt(dx*dx + dy*dy + dz*dz)
        return rr

    def MassEnclosed(self, ptype, radius):
        """
        Compute the enclosed mass for a specific ptype (1=Halo,2=Disk,3=Bulge)
        within a given radius (or array of radii).

        PARAMETERS
        ----------
        ptype : int
            Particle type (1,2,3)
        radius : float or array-like
            Single radius in kpc, or an array of radii in kpc

        RETURNS
        -------
        Menc : float or np.ndarray
            The enclosed mass in Msun, same shape as 'radius'
        """
        # Filter the data for just that ptype
        idx = np.where(self.data['type'] == ptype)[0]
        mass_arr = self.data['m'][idx]  # in 1e10 Msun
        x_arr = self.data['x'][idx]
        y_arr = self.data['y'][idx]
        z_arr = self.data['z'][idx]

        rr = self._distance_from_com(x_arr, y_arr, z_arr)

        # interpret 'radius' to handle both float or array
        r_array = np.atleast_1d(radius)

        # loop over each radius in r_array
        Menc_list = []
        for rmax in r_array:
            # find all particles with rr < rmax
            inside_idx = np.where(rr < rmax)[0]
            # sum the mass (mass is in 1e10 Msun, so final is 1e10 Msun)
            mass_enclosed = np.sum(mass_arr[inside_idx]) * 1e10  # Msun
            Menc_list.append(mass_enclosed)

        if len(Menc_list) == 1:
            return Menc_list[0]
        else:
            return np.array(Menc_list)

    def MassEnclosedTotal(self, radius):
        """
        Sum of halo, disk, and bulge within 'radius'.
        (We assume ptype=1,2,3 are the only ones.)

        RETURNS
        -------
        Mtot : float or array, Msun
        """
        M1 = self.MassEnclosed(1, radius)  # halo
        M2 = self.MassEnclosed(2, radius)  # disk
        M3 = self.MassEnclosed(3, radius)  # bulge if it exists
        return M1 + M2 + M3

def compute_jacobi_radius(r_m31_m33, M_m33, Menc_m31):
    """
    Compute the Jacobi Radius:
      R_J = r_m31_m33 * (M_m33 / (2 * Menc_m31))^(1/3)

    PARAMETERS
    ----------
    r_m31_m33 : float
        The separation between M31 and M33 (kpc)
    M_m33 : float
        M33's total mass (Msun)
    Menc_m31 : float
        M31's enclosed mass at radius r_m31_m33 (Msun)

    RETURNS
    -------
    RJ : float
        The Jacobi radius in kpc
    """
    if Menc_m31 <= 0:
        return 0.0
    RJ = r_m31_m33 * (M_m33 / (2.0 * Menc_m31))**(1.0/3.0)
    return RJ

class MassProfile:
    """
    A class that computes the mass profile for a snapshot, given the galaxy's
    COM position. We can compute the enclosed mass for each or all components.
    """
    def __init__(self, data, xCOM, yCOM, zCOM):
        """
        PARAMETERS
        ----------
        data : np.ndarray
            Snapshot data array with columns 'type', 'm', 'x', 'y', 'z' ...
        xCOM, yCOM, zCOM : float
            The galaxy's center-of-mass in kpc
        """
        self.data = data
        self.xCOM = xCOM
        self.yCOM = yCOM
        self.zCOM = zCOM
        self.G = 4.498768e-6  # gravitational constant in kpc (km/s)^2 / Msun

    def _distance_from_com(self, x, y, z):
        # Return array of distances from (xCOM,yCOM,zCOM)
        dx = x - self.xCOM
        dy = y - self.yCOM
        dz = z - self.zCOM
        rr = np.sqrt(dx**2 + dy**2 + dz**2)
        return rr

    def MassEnclosed(self, ptype, radius):
        """
        Mass of given ptype (1=halo,2=disk,3=bulge) inside radius (kpc).

        If 'radius' is an array, returns an array of enclosed masses.
        Mass is in Msun.
        """
        idx = np.where(self.data['type'] == ptype)[0]
        mass_arr = self.data['m'][idx]  # in 1e10 Msun
        x_arr = self.data['x'][idx]
        y_arr = self.data['y'][idx]
        z_arr = self.data['z'][idx]

        rr = self._distance_from_com(x_arr, y_arr, z_arr)

        # handle single or array radius
        r_array = np.atleast_1d(radius)
        Menc_list = []
        for rmax in r_array:
            inside_idx = np.where(rr < rmax)[0]
            # sum mass, convert from 1e10 Msun to Msun
            mass_enclosed = np.sum(mass_arr[inside_idx]) * 1e10
            Menc_list.append(mass_enclosed)

        if len(Menc_list) == 1:
            return Menc_list[0]
        else:
            return np.array(Menc_list)

    def MassEnclosedTotal(self, radius):
        """
        Sum of ptype=1,2,3 within radius. Returns Msun.
        """
        M1 = self.MassEnclosed(1, radius)
        M2 = self.MassEnclosed(2, radius)
        M3 = self.MassEnclosed(3, radius)
        return M1 + M2 + M3

# The main demonstration function that loads All_COM.csv, loops over
# snapshots, reads M31 & M33 snapshot data, and computes Jacobi radius

def demo_jacobi_usage():
    """
    1) Load COM data from All_COM.csv
    2) For each snapshot, read M31/M33 data
    3) Build MassProfile for M31
    4) Compute r_M31_M33, Menc(M31), M33 total mass, then Jacobi radius
    5) Save results to JacobiRadius.csv
    """

    # 1) read the entire COM CSV
    com_df = pd.read_csv("All_COM.csv")

    # create a new column 'snap_int'
    try:
        com_df['snap_int'] = com_df['snapshot'].astype(int)
    except ValueError:
        pass
    if 'snap_int' in com_df.columns:
        all_snaps = sorted(com_df['snap_int'].unique())
    else:
        all_snaps = sorted(com_df['snapshot'].unique())

    jaco_records = []

    def get_com_row(gal, snap, ptype):
        if 'snap_int' in com_df.columns:
            row_sel = com_df[(com_df['galaxy'] == gal)
                             & (com_df['ptype'] == ptype)
                             & (com_df['snap_int'] == snap)]
        else:
            snap_str = str(snap).zfill(3)  # zero-pad to 3 digits
            row_sel = com_df[(com_df['galaxy']==gal)&
                             (com_df['ptype']==ptype)&
                             (com_df['snapshot']==snap_str)]
        if len(row_sel) == 1:
            return row_sel.iloc[0]
        else:
            return None

    # 2) iterate over snapshot numbers
    for snap in all_snaps:
        snap_str = f"{int(snap):03d}"

        m31_filename = f"M31/M31_{snap_str}.txt"
        m33_filename = f"M33/M33_{snap_str}.txt"

        # read data
        if not os.path.exists(m31_filename) or not os.path.exists(m33_filename):
            # if either file is missing, skip
            continue

        time_myr_m31, _, data_m31 = Read(m31_filename)
        time_myr_m33, _, data_m33 = Read(m33_filename)

        # get M31's disk COM row
        row_m31 = get_com_row("M31", snap, 2)
        if row_m31 is None:
            continue
        m31_xcom = row_m31['xcom']
        m31_ycom = row_m31['ycom']
        m31_zcom = row_m31['zcom']
        # use time from that row
        time_myr = row_m31['time_Myr']

        # get M33's disk COM row
        row_m33 = get_com_row("M33", snap, 2)
        if row_m33 is None:
            continue
        m33_xcom = row_m33['xcom']
        m33_ycom = row_m33['ycom']
        m33_zcom = row_m33['zcom']

        # distance
        dx = m33_xcom - m31_xcom
        dy = m33_ycom - m31_ycom
        dz = m33_zcom - m31_zcom
        r_m31_m33 = np.sqrt(dx*dx + dy*dy + dz*dz)

        # Build a MassProfile for M31
        m31prof = MassProfile(data_m31, m31_xcom, m31_ycom, m31_zcom)
        Menc_m31 = m31prof.MassEnclosedTotal(r_m31_m33)

        # total M33 mass
        m33prof = MassProfile(data_m33, m33_xcom, m33_ycom, m33_zcom)
        M33_total = m33prof.MassEnclosedTotal(300.0)

        # compute RJ
        RJ = compute_jacobi_radius(r_m31_m33, M33_total, Menc_m31)

        # store
        rec = {
            "snapshot": snap_str,
            "snap_int": snap,
            "time_Myr": time_myr,
            "r_M31_M33": r_m31_m33,
            "M31_enc": Menc_m31,
            "M33_total": M33_total,
            "JacobiR": RJ
        }
        jaco_records.append(rec)

    # done looping
    jaco_df = pd.DataFrame(jaco_records)
    jaco_df.to_csv("JacobiRadius.csv", index=False)
    print(f"Saved {len(jaco_df)} rows to JacobiRadius.csv.")


# now run
demo_jacobi_usage()

df_jacobi = pd.read_csv("JacobiRadius.csv")
plt.plot(df_jacobi["time_Myr"], df_jacobi["JacobiR"], '-')
plt.xlabel("Time (Myr)")
plt.ylabel("Jacobi Radius (kpc)")
plt.title("Evolution of Jacobi Radius Over Time")
plt.show()

import pandas as pd
import numpy as np
import os

def mass_loss_m33_demo():
    """
    1) Load JacobiRadius.csv (which has M33's R_J for each snapshot)
    2) For each snapshot:
       - Read M33's snapshot file
       - Find M33's COM from All_COM.csv (ptype=2 or disk)
       - Compute the total 'stellar' mass inside R_J
       - Compare to the initial snapshot's stellar mass to get fraction
    3) Save results to M33_MassLoss.csv
    """
    jaco_df = pd.read_csv("JacobiRadius.csv")
    com_df = pd.read_csv("All_COM.csv")

    m33_com_df = com_df[(com_df['galaxy']=="M33") & (com_df['ptype']==2)]

    # interpret snapshots as integers so we can match them with zero-padded filenames
    def parse_snapshot_as_int(s):
        try:
            return int(s)
        except:
            return None

    # create a 'snap_int' column in both DataFrames
    if 'snap_int' not in jaco_df.columns:
        jaco_df['snap_int'] = jaco_df['snapshot'].apply(parse_snapshot_as_int)
    if 'snap_int' not in m33_com_df.columns:
        m33_com_df['snap_int'] = m33_com_df['snapshot'].apply(parse_snapshot_as_int)

    # We'll define a small function that, given a snapshot number, returns:
    #   1) The R_J from jacobi_df
    #   2) M33's COM from m33_com_df
    #   3) The time from either table
    def get_jacobi_and_com(snap_int):
        # find the row in jaco_df
        row_j = jaco_df[jaco_df['snap_int'] == snap_int]
        if len(row_j) != 1:
            return None
        RJ = row_j.iloc[0]['JacobiR']
        time_myr = row_j.iloc[0]['time_Myr']

        row_c = m33_com_df[m33_com_df['snap_int'] == snap_int]
        if len(row_c) != 1:
            return None
        xcom = row_c.iloc[0]['xcom']
        ycom = row_c.iloc[0]['ycom']
        zcom = row_c.iloc[0]['zcom']

        return (time_myr, RJ, xcom, ycom, zcom)

    # We'll define "initial mass" as the total star mass (disk+bulge)
    # inside e.g. 300 kpc at snapshot=0. We'll compute that once.
    # We'll pick the total star mass for simplicity.

    # We'll define a helper function to get "stellar mass" for M33
    # either disk+bulge = ptype=2,3 or if you prefer only ptype=2, etc.
    def get_stellar_mass(filename, xCOM, yCOM, zCOM):
        # We'll read the snapshot, then sum ptype=2 and 3 inside some big radius
        timeMyr, total, data = Read(filename)
        # We'll define a function to compute distance
        dx = data['x'] - xCOM
        dy = data['y'] - yCOM
        dz = data['z'] - zCOM
        rr = np.sqrt(dx**2 + dy**2 + dz**2)

        # We'll define star types as ptype=2 or 3
        star_idx = np.where((data['type']==2) | (data['type']==3))[0]

        # sum all star mass from star_idx
        # mass is in 1e10 Msun, so multiply by 1e10
        star_mass = np.sum(data['m'][star_idx]) * 1e10
        return star_mass

    # Step: find the initial snapshot row in jaco_df
    # which is snap_int=0 presumably
    initial_row = jaco_df[jaco_df['snap_int']==0]
    if len(initial_row)!=1:
        # if there's no snapshot=0 in JacobiRadius, we might fallback
        print("Warning: No snapshot=0 in JacobiRadius.csv. Will attempt next best.")
        # fallback approach
        test_snaps = sorted(jaco_df['snap_int'].dropna().unique())
        if len(test_snaps)==0:
            print("No snapshots found in jaco_df. Exiting.")
            return
        zero_snap = test_snaps[0]
    else:
        zero_snap = 0

    # read M33_000.txt for the 'initial' star mass
    m33_init_file = f"M33/M33_{zero_snap:03d}.txt"
    # retrieve the COM from get_jacobi_and_com if available
    initval = get_jacobi_and_com(zero_snap)
    if initval is None:
        print("No valid data for snapshot=0. Cannot define initial mass.")
        return
    # parse
    t0, RJ0, x0com, y0com, z0com = initval
    # get the "initial star mass" => sum (ptype=2,3) at t=0
    M33_init_star = get_stellar_mass(m33_init_file, x0com, y0com, z0com)
    print(f"Initial M33 star mass at snapshot=0 = {M33_init_star:.2e} Msun")

    # Now we loop over all snapshots in jaco_df
    bound_records = []
    for snap in sorted(jaco_df['snap_int'].dropna().unique()):
        # get the data we need
        result = get_jacobi_and_com(snap)
        if result is None:
            continue
        time_myr, RJ, xcom, ycom, zcom = result

        # build the M33 filename
        snap_str = f"{int(snap):03d}"
        m33_file = f"M33/M33_{snap_str}.txt"
        if not os.path.exists(m33_file):
            # skip if missing
            continue

        # We'll read M33, then sum star mass inside R_J
        timeM, tot, dataM = Read(m33_file)
        # compute distance from M33 COM
        dx = dataM['x'] - xcom
        dy = dataM['y'] - ycom
        dz = dataM['z'] - zcom
        rr = np.sqrt(dx**2 + dy**2 + dz**2)

        # star types: ptype=2 or 3
        star_idx = np.where((dataM['type']==2)|(dataM['type']==3))[0]
        # among those star_idx, who is inside R_J
        inside_idx = star_idx[np.where(rr[star_idx] < RJ)]
        # sum mass
        Mstar_bound = np.sum(dataM['m'][inside_idx]) * 1e10  # Msun
        frac_bound = Mstar_bound / M33_init_star

        rec = {
            "snapshot": snap_str,
            "snap_int": snap,
            "time_Myr": time_myr,
            "JacobiR": RJ,
            "Mstar_bound": Mstar_bound,
            "frac_bound": frac_bound
        }
        bound_records.append(rec)

    # done
    df_bound = pd.DataFrame(bound_records)
    df_bound.to_csv("M33_MassLoss.csv", index=False)
    print(f"Mass loss results saved to M33_MassLoss.csv with {len(df_bound)} rows.")


# Now let's execute the demonstration
mass_loss_m33_demo()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_m33_mass_loss():
    df_loss = pd.read_csv("M33_MassLoss.csv")
    # Columns assumed: ["snapshot", "snap_int", "time_Myr", "JacobiR", "Mstar_bound", "frac_bound"]

    if len(df_loss) == 0:
        print("No data in M33_MassLoss.csv. Exiting.")
        return

    initial_mass = df_loss['Mstar_bound'].iloc[0] / df_loss['frac_bound'].iloc[0]  # Msun
    final_mass   = df_loss['Mstar_bound'].iloc[-1]  # Msun
    frac_lost    = 1.0 - df_loss['frac_bound'].iloc[-1]

    init_str  = f"{initial_mass:.2e} Msun"
    final_str = f"{final_mass:.2e} Msun"
    lost_str  = f"{(frac_lost*100):.1f}%"

    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(df_loss["time_Myr"], df_loss["frac_bound"], '-', label="M33 Bound Fraction")

    legend_title = f"Initial Mass = {init_str}\nFinal Mass = {final_str}\nLost = {lost_str}"

    ax.legend(title=legend_title, loc="best")

    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel("Fraction of M33 Stellar Mass Bound")
    ax.set_title("M33 Mass Loss Over Time")

    # 9) Show or save
    plt.show()

# run
plot_m33_mass_loss()

# 1) ROTATE TO FACE-ON
def rotate_to_face_on(positions, velocities):
    """
    Rotate M33's disk to face-on coordinates so that its angular momentum
    vector lies along the z-axis and the disk is in the XY-plane.
    
    PARAMETERS
    ----------
    positions : (N,3) array
        x,y,z for each star particle
    velocities : (N,3) array
        vx,vy,vz for each star particle

    RETURNS
    -------
    new_pos : (N,3) array
        Rotated positions
    new_vel : (N,3) array
        Rotated velocities
    """
    # 1) compute total angular momentum
    #    L = sum( r cross v )
    L = np.array([0.0, 0.0, 0.0])
    for i in range(len(positions)):
        r = positions[i]
        v = velocities[i]
        L += np.cross(r,v)
    
    # normalize
    Lmag = np.sqrt(L.dot(L))
    if Lmag==0:
        # if zero, can't define orientation; just return original
        return positions, velocities
    Lhat = L / Lmag
    
    # define z unit vector
    zhat = np.array([0.0, 0.0, 1.0])
    
    # cross to get axis
    vv = np.cross(Lhat, zhat)
    s = np.linalg.norm(vv)
    c = np.dot(Lhat, zhat)  # Lhat dot zhat
    # if s=0, means Lhat parallel or anti-parallel to z
    if s==0:
        # already aligned
        return positions, velocities
    
    # define rotation matrix using Rodrigues’ rotation formula
    # R = I + [v_x] + [v_x]^2 * (1 - c)/s^2
    vx = np.array([
        [0.0, -vv[2], vv[1]],
        [vv[2], 0.0, -vv[0]],
        [-vv[1], vv[0], 0.0]
    ])
    I = np.eye(3)
    vx2 = vx.dot(vx)
    R = I + vx + vx2*((1.0-c)/(s*s))
    
    # apply rotation
    new_pos = positions.dot(R.T)
    new_vel = velocities.dot(R.T)
    return new_pos, new_vel

# 2) SURFACE DENSITY PROFILE
def surface_density_profile(positions, nbins=50, rmax=None):
    """
    Compute the radial surface density profile for a face-on disk projection.

    PARAMETERS
    ----------
    positions : (N,3) array
        face-on coordinates of star particles => disk in xy-plane
    nbins : int
        number of radial bins
    rmax : float or None
        max radius to consider; if None, pick the 99 percentile or so

    RETURNS
    -------
    r_mid : array
        The midpoints of the radial bins
    sigma : array
        The surface density in each bin
    """
    # radial distance in the XY plane
    x = positions[:,0]
    y = positions[:,1]
    r = np.sqrt(x*x + y*y)

    # decide rmax if not given
    if rmax is None:
        rmax = np.percentile(r, 99.0)  # near max but ignoring outliers

    # define radial bins
    edges = np.linspace(0, rmax, nbins+1)
    r_mid = 0.5*(edges[1:] + edges[:-1])
    sigma = np.zeros(nbins)
    
    # area of each annulus = pi (r_outer^2 - r_inner^2)
    
    for i in range(nbins):
        r_in = edges[i]
        r_out = edges[i+1]
        area = np.pi*(r_out**2 - r_in**2)
        # select those in this annulus
        idx = np.where((r>=r_in)&(r<r_out))[0]
        # surface density = (number of particles) / area
        # (if you want mass-based, you'd sum mass - but let's do number-based for demonstration)
        # For star mass-based approach, you'd pass an array of masses instead of just positions
        # We'll just do "count" here, but you can easily adapt.
        count = len(idx)
        sigma[i] = count / area
    
    return r_mid, sigma

# 3) SERSIC / EXPONENTIAL FIT
def sersic_function(r, I0, re, n):
    """
    Basic Sersic profile:
    I(r) = I0 * exp( -b*( (r/re)^(1/n) - 1 ) )
    where b ~ 2n - 1/3 + ...
    For simplicity, let's define b = 1.9992 n - 0.3271 for n>0.5 approximate
    This is a rough approximation.
    """
    # approximate b(n)
    b = 2.0*n - 1.0/3.0
    x = (r/re)**(1.0/n)
    return I0 * np.exp( -b*( x - 1.0 ) )

def exponential_function(r, I0, r_d):
    """
    Simple exponential disk:
    I(r) = I0 * exp(- r / r_d)
    """
    return I0 * np.exp(-r/r_d)

def fit_sersic(r, sigma, guess=(1.0, 1.0, 1.0), use_exponential=False):
    """
    Fit either a Sersic or an exponential to the radial profile (r, sigma).
    
    PARAMETERS
    ----------
    r : array
    sigma : array
    guess : tuple
        initial guess for (I0, re, n) or (I0, r_d)
    use_exponential : bool
        if True, fit exponential_function, else sersic_function

    RETURNS
    -------
    popt, pcov : from curve_fit
    """
    from scipy.optimize import curve_fit

    # We'll do a simple mask to ignore zero or negative sigma
    idx = np.where(sigma>0)[0]
    r_fit = r[idx]
    y_fit = sigma[idx]

    if len(r_fit)<3:
        # not enough points
        return None, None

    if use_exponential:
        try:
            popt, pcov = curve_fit(exponential_function, r_fit, y_fit, p0=guess[:2])
        except:
            popt, pcov = (None, None)
    else:
        try:
            popt, pcov = curve_fit(sersic_function, r_fit, y_fit, p0=guess)
        except:
            popt, pcov = (None,None)
    return popt, pcov

# 4) DEMONSTRATION: SHIFT & ROTATE M33, THEN 2D HISTOGRAM, THEN FIT
def disk_profile():
    """
    We'll:
      1) Choose a list of snapshots to analyze for M33
      2) For each snapshot, read data, keep star ptypes (2,3), shift by M33's COM
      3) Rotate so disk is face-on
      4) Make a 2D hist, produce radial profile, do a sersic or exponential fit
      5) Save or display a figure, store fit parameters in a CSV
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # We'll read from "All_COM.csv" to find M33's COM
    com_df = pd.read_csv("All_COM.csv")
    # just M33 disk
    m33_df = com_df[(com_df['galaxy']=="M33")&(com_df['ptype']==2)]
    # parse snap as int if needed
    if 'snap_int' not in m33_df.columns:
        m33_df['snap_int'] = m33_df['snapshot'].astype(int)
    # pick a set of snapshots
    test_snaps = [0, 50, 100]  # or any list you want
    records = []

    for snap in test_snaps:
        # find row
        row_sel = m33_df[m33_df['snap_int']==snap]
        if len(row_sel)!=1:
            continue
        row = row_sel.iloc[0]
        xcom = row['xcom']
        ycom = row['ycom']
        zcom = row['zcom']
        time_myr = row['time_Myr']

        # build file name
        snap_str = f"{snap:03d}"
        fname = f"M33/M33_{snap_str}.txt"
        if not os.path.exists(fname):
            continue
        # read
        tm, tp, data = Read(fname)
        # filter for star ptypes
        idx_star = np.where( (data['type']==2)|(data['type']==3) )[0]
        # positions
        x = data['x'][idx_star] - xcom
        y = data['y'][idx_star] - ycom
        z = data['z'][idx_star] - zcom
        # velocities
        vx = data['vx'][idx_star]
        vy = data['vy'][idx_star]
        vz = data['vz'][idx_star]

        # build arrays shape(N,3)
        pos_arr = np.column_stack((x,y,z))
        vel_arr = np.column_stack((vx,vy,vz))

        # rotate to face-on
        new_pos, new_vel = rotate_to_face_on(pos_arr, vel_arr)

        # we can do a 2D hist for a figure
        fig,ax = plt.subplots(1,1, figsize=(8,6))
        # new_pos[:,0] => x', new_pos[:,1] => y'
        ax.hist2d(new_pos[:,0], new_pos[:,1], bins=200, cmap='magma', norm=None)
        ax.set_xlabel("X' (kpc)")
        ax.set_ylabel("Y' (kpc)")
        ax.set_title(f"M33 Face-On, snap={snap_str}, t={time_myr:.1f} Myr")
        #plt.savefig(f"M33_FaceOn_{snap_str}.png")
        plt.close(fig)

        # get radial surface density
        r_mid, sigma = surface_density_profile(new_pos, nbins=40, rmax=20.0)
        # fit an exponential
        popt_exp, pcov_exp = fit_sersic(r_mid, sigma, guess=(np.max(sigma), 2.0), use_exponential=True)
        # fit a sersic
        popt_ser, pcov_ser = fit_sersic(r_mid, sigma, guess=(np.max(sigma), 2.0, 1.0), use_exponential=False)

        # store results
        rec = {
            "snapshot": snap,
            "time_Myr": time_myr
        }
        # exponential popt => I0, r_d
        if popt_exp is not None:
            rec["exp_I0"] = popt_exp[0]
            rec["exp_r_d"] = popt_exp[1]
        else:
            rec["exp_I0"] = 0
            rec["exp_r_d"] = 0
        # sersic popt => I0, re, n
        if popt_ser is not None and len(popt_ser)==3:
            rec["sersic_I0"] = popt_ser[0]
            rec["sersic_re"] = popt_ser[1]
            rec["sersic_n"]  = popt_ser[2]
        else:
            rec["sersic_I0"] = 0
            rec["sersic_re"] = 0
            rec["sersic_n"]  = 0

        records.append(rec)

    # end for snap
    df_out = pd.DataFrame(records)
    df_out.to_csv("M33_DiskProfileFits.csv", index=False)
    print(f"Done. Wrote {len(df_out)} rows to M33_DiskProfileFits.csv.")

# Let's run it:
disk_profile()


def rotate_to_face_on(positions, velocities):
    """
    Rotate a disk to face-on by aligning the total angular momentum vector with z-axis.
    positions, velocities : (N,3) arrays
    returns new_pos, new_vel (N,3) arrays
    """
    L = np.array([0.0,0.0,0.0])
    for i in range(len(positions)):
        r = positions[i]
        v = velocities[i]
        L += np.cross(r,v)
    Lmag = np.linalg.norm(L)
    if Lmag==0:
        return positions, velocities
    Lhat = L / Lmag

    zhat = np.array([0,0,1])
    vv = np.cross(Lhat, zhat)
    s = np.linalg.norm(vv)
    c = np.dot(Lhat, zhat)
    if s==0:
        # already aligned
        return positions, velocities
    
    vx = np.array([
        [0.0, -vv[2], vv[1]],
        [vv[2],  0.0, -vv[0]],
        [-vv[1], vv[0], 0.0]
    ])
    I = np.eye(3)
    vx2 = vx.dot(vx)
    R = I + vx + vx2*((1.0-c)/s**2)

    new_pos = positions.dot(R.T)
    new_vel = velocities.dot(R.T)
    return new_pos, new_vel

def surface_density_profile(positions, nbins=50, rmax=None):
    """
    Create a radial surface density profile (number-based, not mass-based).
    positions: (N,3), face-on coordinates => disk in xy-plane
    nbins: number of radial bins
    rmax: if None, pick ~99 percentile of radius
    returns r_mid, sigma arrays
    """
    x = positions[:,0]
    y = positions[:,1]
    r = np.sqrt(x*x + y*y)

    if rmax is None:
        rmax = np.percentile(r, 99.0)
    edges = np.linspace(0, rmax, nbins+1)
    r_mid = 0.5*(edges[1:] + edges[:-1])
    sigma = np.zeros(nbins)
    
    for i in range(nbins):
        r_in = edges[i]
        r_out = edges[i+1]
        area = np.pi*(r_out**2 - r_in**2)
        idx = np.where((r>=r_in)&(r<r_out))[0]
        count = len(idx)
        sigma[i] = count/area
    
    return r_mid, sigma

def sersic_function(r, I0, re, n):
    """
    Sersic: I(r) = I0 * exp{ -b*( (r/re)^(1/n) - 1 ) },
    with b ~ 2n - 1/3 (approx).
    """
    b = 2.0*n - 1.0/3.0
    x = (r/re)**(1.0/n)
    return I0*np.exp(-b*(x-1.0))

def exponential_function(r, I0, rd):
    """
    Exponential disk: I(r) = I0 * exp(-r/rd)
    """
    return I0*np.exp(-r/rd)

def fit_sersic(r, sigma, guess=(1.0, 1.0, 1.0), use_exponential=False):
    """
    Fit either a Sersic or exponential to (r, sigma).
    If use_exponential=True => I(r)=I0 exp(-r/rd)
    Otherwise => Sersic
    """
    idx = np.where(sigma>0)[0]
    r_fit = r[idx]
    y_fit = sigma[idx]
    if len(r_fit)<3:
        return None, None
    try:
        if use_exponential:
            popt, pcov = curve_fit(exponential_function, r_fit, y_fit, p0=guess[:2])
        else:
            popt, pcov = curve_fit(sersic_function, r_fit, y_fit, p0=guess)
    except:
        popt, pcov = (None,None)
    return popt, pcov

# MAIN FUNCTION
def disk_profile():
    """
    We'll read M33's COM data from 'All_COM.csv' for ptype=2,
    loop over all snapshots, read M33_xxx.txt, center on COM, rotate
    face-on, compute radial surface density, fit exponential + sersic,
    produce face-on plots, store results in 'M33_DiskProfileFits.csv'.
    """
    com_df = pd.read_csv("All_COM.csv")
    # filter for M33 disk
    m33_df = com_df[(com_df['galaxy']=="M33") & (com_df['ptype']==2)].copy()
    # parse snapshot as int if possible
    if 'snap_int' not in m33_df.columns:
        m33_df['snap_int'] = m33_df['snapshot'].astype(int)

    # We'll collect results
    records = []

    # sort snapshots
    snap_list = sorted(m33_df['snap_int'].unique())

    for snap in snap_list:
        row_sel = m33_df[m33_df['snap_int']==snap]
        if len(row_sel)!=1:
            continue
        row = row_sel.iloc[0]
        xcom = row['xcom']
        ycom = row['ycom']
        zcom = row['zcom']
        time_myr = row['time_Myr']

        snap_str = f"{snap:03d}"
        fname = f"M33/M33_{snap_str}.txt"
        if not os.path.exists(fname):
            continue

        # read file
        tm, tot, data = Read(fname)
        # pick star particles
        idx_star = np.where((data['type']==2)|(data['type']==3))[0]
        x = data['x'][idx_star] - xcom
        y = data['y'][idx_star] - ycom
        z = data['z'][idx_star] - zcom
        vx = data['vx'][idx_star]
        vy = data['vy'][idx_star]
        vz = data['vz'][idx_star]

        pos_arr = np.column_stack((x,y,z))
        vel_arr = np.column_stack((vx,vy,vz))

        # rotate
        new_pos, new_vel = rotate_to_face_on(pos_arr, vel_arr)

        # produce face-on 2D hist
        fig, ax = plt.subplots(figsize=(8,6))
        ax.hist2d(new_pos[:,0], new_pos[:,1], bins=200, cmap='magma')
        ax.set_title(f"M33 Face-On snapshot={snap_str}, t={time_myr:.1f} Myr")
        ax.set_xlabel("X' (kpc)")
        ax.set_ylabel("Y' (kpc)")
        #plt.savefig(f"M33_FaceOn_{snap_str}.png")
        plt.close(fig)

        # radial surface density
        r_mid, sigma = surface_density_profile(new_pos, nbins=40, rmax=20.0)

        # fit exponential
        popt_exp, pcov_exp = fit_sersic(r_mid, sigma, guess=(np.max(sigma), 2.0), use_exponential=True)
        # fit sersic
        popt_ser, pcov_ser = fit_sersic(r_mid, sigma, guess=(np.max(sigma), 2.0, 1.0), use_exponential=False)

        rec = {
            "snapshot": snap,
            "time_Myr": time_myr
        }
        if popt_exp is not None:
            # popt_exp => I0, r_d
            rec["exp_I0"] = popt_exp[0]
            rec["exp_r_d"] = popt_exp[1]
        else:
            rec["exp_I0"] = 0
            rec["exp_r_d"] = 0

        if popt_ser is not None and len(popt_ser)==3:
            rec["sersic_I0"] = popt_ser[0]
            rec["sersic_re"] = popt_ser[1]
            rec["sersic_n"]  = popt_ser[2]
        else:
            rec["sersic_I0"] = 0
            rec["sersic_re"] = 0
            rec["sersic_n"]  = 0

        records.append(rec)

    df_out = pd.DataFrame(records)
    df_out.to_csv("M33_DiskProfileFits.csv", index=False)
    print(f"Created M33_DiskProfileFits.csv with {len(df_out)} rows.")

disk_profile()

def compute_disk_kinematics(new_pos, new_vel, nbins=20, rmax=None):
    """
    Given face-on disk coordinates (new_pos) & velocities (new_vel),
    compute radial bins out to rmax, and measure velocity dispersions
    (sigma_rad, sigma_tan, sigma_z) in each bin.

    PARAMETERS
    ----------
    new_pos : (N,3) array
        Face-on coordinates for star particles
    new_vel : (N,3) array
        Corresponding face-on velocities
    nbins : int
        Number of radial bins
    rmax : float or None
        Maximum radius in kpc to consider; if None, pick ~99th percentile

    RETURNS
    -------
    r_mid : array of shape (nbins,)
    sigma_rad : array of shape (nbins,)
    sigma_tan : array of shape (nbins,)
    sigma_z   : array of shape (nbins,)
    """

    x = new_pos[:,0]
    y = new_pos[:,1]
    z = new_pos[:,2]
    vx = new_vel[:,0]
    vy = new_vel[:,1]
    vz = new_vel[:,2]

    # compute cylindrical radius in the face-on plane
    r = np.sqrt(x**2 + y**2)
    # define phi = arctan2(y, x)
    phi = np.arctan2(y, x)
    # radial velocity v_rad = vx cos(phi) + vy sin(phi)
    v_rad = vx*np.cos(phi) + vy*np.sin(phi)
    # tangential velocity v_tan = -vx sin(phi) + vy cos(phi)
    # (this is the direction 90 deg from radial)
    v_tan = -vx*np.sin(phi) + vy*np.cos(phi)
    # vertical velocity is just vz in face-on coordinates

    if rmax is None:
        rmax = np.percentile(r, 99.0)

    edges = np.linspace(0, rmax, nbins+1)
    r_mid = 0.5*(edges[:-1]+edges[1:])
    sigma_rad = np.zeros(nbins)
    sigma_tan = np.zeros(nbins)
    sigma_z   = np.zeros(nbins)

    for i in range(nbins):
        rin = edges[i]
        rout = edges[i+1]
        idx  = np.where((r>=rin)&(r<rout))[0]
        if len(idx)<2:
            # not enough points
            sigma_rad[i] = 0
            sigma_tan[i] = 0
            sigma_z[i]   = 0
            continue
        # measure standard deviations
        sigma_rad[i] = np.std(v_rad[idx])
        sigma_tan[i] = np.std(v_tan[idx])
        sigma_z[i]   = np.std(vz[idx])

    return r_mid, sigma_rad, sigma_tan, sigma_z


def disk_kinematics_demo():
    """
    For each M33 snapshot in All_COM.csv (ptype=2), read M33 data, rotate face-on,
    compute velocity dispersions in radial bins, store in CSV.

    We'll produce a single CSV "M33_Kinematics.csv" that has, for each snapshot,
    the radial bin midpoints plus the velocity dispersions in each bin. We'll do
    a wide-format approach: e.g. col = (snapshot, r1, sigma_rad1, sigma_tan1, sigma_z1, r2,...).
    Alternatively, we can do a long-format where each bin is a row.
    Below we'll do a simple *long-format* approach with columns:
        snapshot, bin_index, r_mid, sigma_rad, sigma_tan, sigma_z
    """

    com_df = pd.read_csv("All_COM.csv")
    # filter for M33 disk
    m33_df = com_df[(com_df['galaxy']=="M33") & (com_df['ptype']==2)].copy()
    if 'snap_int' not in m33_df.columns:
        m33_df['snap_int'] = m33_df['snapshot'].astype(int)

    snap_list = sorted(m33_df['snap_int'].unique())
    records = []

    for snap in snap_list:
        row_sel = m33_df[m33_df['snap_int']==snap]
        if len(row_sel)!=1:
            continue
        row = row_sel.iloc[0]
        xcom = row['xcom']
        ycom = row['ycom']
        zcom = row['zcom']
        time_myr = row['time_Myr']

        snap_str = f"{snap:03d}"
        fname = f"M33/M33_{snap_str}.txt"
        if not os.path.exists(fname):
            continue

        # read
        tm, tot, data = Read(fname)
        # pick star ptypes
        idx_star = np.where((data['type']==2)|(data['type']==3))[0]
        x = data['x'][idx_star] - xcom
        y = data['y'][idx_star] - ycom
        z = data['z'][idx_star] - zcom
        vx = data['vx'][idx_star]
        vy = data['vy'][idx_star]
        vz = data['vz'][idx_star]

        pos_arr = np.column_stack((x,y,z))
        vel_arr = np.column_stack((vx,vy,vz))

        # rotate face-on
        from math import isfinite

        new_pos, new_vel = rotate_to_face_on(pos_arr, vel_arr)

        # measure velocity dispersions in radial bins
        r_mid, sigma_rad, sigma_tan, sigma_z = compute_disk_kinematics(new_pos, new_vel, nbins=20, rmax=20.0)

        # store in "long format"
        for i in range(len(r_mid)):
            # handle any possible non-finite results
            sr = sigma_rad[i]
            st = sigma_tan[i]
            sz = sigma_z[i]
            if not isfinite(sr): sr=0
            if not isfinite(st): st=0
            if not isfinite(sz): sz=0
            rec = {
                "snapshot": snap,
                "time_Myr": time_myr,
                "bin_index": i,
                "r_mid": r_mid[i],
                "sigma_rad": sr,
                "sigma_tan": st,
                "sigma_z"  : sz
            }
            records.append(rec)

    df_out = pd.DataFrame(records)
    df_out.to_csv("M33_Kinematics.csv", index=False)
    print(f"disk_kinematics_demo finished. Wrote {len(df_out)} rows to M33_Kinematics.csv.")

disk_kinematics_demo()

df_kin = pd.read_csv("M33_Kinematics.csv")
df_100 = df_kin[df_kin['snapshot']==100]
plt.plot(df_100['r_mid'], df_100['sigma_z'], 'o-')
plt.xlabel("Radius (kpc)")
plt.ylabel("Vertical Velocity Dispersion (km/s)")
plt.title("M33 Kinematics at snapshot=100")
plt.show()

def final_reporting():
    """
    Creates final plots/tables for the paper:
      1) M33 mass fraction vs time (from M33_MassLoss.csv)
      2) M33 disk profile fits (from M33_DiskProfileFits.csv)
      3) M33 disk velocity dispersions, but one figure per snapshot (from M33_Kinematics.csv)
      4) Summaries saved in a text/csv in 'figures/'.

    Adjust as needed for your final suite of results.
    """

    ########################################################################
    # 0) Make a 'figures' folder if not present
    ########################################################################
    if not os.path.exists("figures"):
        os.mkdir("figures")

    ########################################################################
    # 1) Plot M33 disk profile fits (M33_DiskProfileFits.csv)
    ########################################################################
    diskfits_file = "M33_DiskProfileFits.csv"
    if os.path.exists(diskfits_file):
        df_fits = pd.read_csv(diskfits_file)
        # typical columns: snapshot, time_Myr, exp_I0, exp_r_d, sersic_I0, sersic_re, sersic_n

        fig, ax = plt.subplots(2, 1, figsize=(12,10), dpi=120)

        # Subplot 1: Exponential scale length
        ax[0].plot(df_fits["time_Myr"], df_fits["exp_r_d"], '-', label='exp_r_d')
        ax[0].set_xlabel("Time (Myr)")
        ax[0].set_ylabel("Exponential Scale Length (kpc)")
        ax[0].set_title("M33 Disk Scale Length Over Time")
        ax[0].legend()

        # Subplot 2: Sersic n
        ax[1].plot(df_fits["time_Myr"], df_fits["sersic_n"], '-', label='sersic_n')
        ax[1].set_xlabel("Time (Myr)")
        ax[1].set_ylabel("Sersic Index n")
        ax[1].set_title("M33 Sersic Index Over Time")
        ax[1].legend()

        plt.tight_layout()
        plt.savefig("M33_DiskProfileFits.png")
        plt.close(fig)
        print("Saved M33_DiskProfileFits.png")

    ########################################################################
    # 2) Disk Kinematics: We handle M33_Kinematics.csv
    #    But we produce a SEPARATE figure for each snapshot => no huge multi-subplot
    ########################################################################
    kin_file = "M33_Kinematics.csv"
    if os.path.exists(kin_file):
        df_kin = pd.read_csv(kin_file)
        # columns: snapshot, time_Myr, bin_index, r_mid, sigma_rad, sigma_tan, sigma_z
        # We'll identify unique snapshots
        unique_snaps = sorted(df_kin["snapshot"].unique())

        for snap in unique_snaps:
            sub = df_kin[df_kin["snapshot"]==snap]
            if len(sub)==0:
                continue
            # We'll take the time from the first row
            time_myr = sub.iloc[0]["time_Myr"]

            fig, ax = plt.subplots(figsize=(6,4), dpi=120)
            ax.plot(sub["r_mid"], sub["sigma_rad"], 'r-o', label=r'$\sigma_{\mathrm{rad}}$')
            ax.plot(sub["r_mid"], sub["sigma_tan"], 'g-o', label=r'$\sigma_{\mathrm{tan}}$')
            ax.plot(sub["r_mid"], sub["sigma_z"],   'b-o', label=r'$\sigma_{z}$')
            ax.set_xlabel("r (kpc)")
            ax.set_ylabel("Velocity Dispersion (km/s)")
            ax.set_title(f"M33 Kinematics, snap={snap}, t={time_myr:.1f} Myr")
            ax.legend()
            plt.tight_layout()

            outname = f"M33_DiskKinematics_{snap}.png"
            #plt.savefig(outname)
            plt.close(fig)
            #plt.show()
            print(f"Saved {outname}")

    ########################################################################
    # 3) Summarize final numeric results in a text or CSV
    ########################################################################
    massloss_file = "M33_MassLoss.csv"
    summary_rows = []
    # example: from massloss_file, we pick the final row
    if os.path.exists(massloss_file):
        df_loss = pd.read_csv(massloss_file)
        last_row_loss = df_loss.iloc[-1]
        summary_rows.append({
            "description":"Final M33 mass fraction",
            "snapshot": last_row_loss["snapshot"],
            "time_Myr": last_row_loss["time_Myr"],
            "value": last_row_loss["frac_bound"]
        })

    # from diskfits_file, final row
    if os.path.exists(diskfits_file):
        df_fits = pd.read_csv(diskfits_file)
        last_row_fit = df_fits.iloc[-1]
        summary_rows.append({
            "description":"Final M33 sersic n",
            "snapshot": last_row_fit["snapshot"],
            "time_Myr": last_row_fit["time_Myr"],
            "value": last_row_fit["sersic_n"]
        })

    if len(summary_rows)>0:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv("Final_Summary.csv", index=False)
        print("Wrote a small summary table to Final_Summary.csv.")
    else:
        print("No final summary to write (missing input files).")

    print("All final plotting/reporting steps completed.")


# Let's call it
final_reporting()


import nbformat

notebook_path = 'Project Code V4.ipynb'

nb = nbformat.read(notebook_path, as_version=4)

total_code_lines = 0

for cell in nb.cells:
    if cell.cell_type == 'code':
        lines = cell.source.splitlines()
        total_code_lines += len(lines)

print(f"Total lines of code in the notebook: {total_code_lines}")


