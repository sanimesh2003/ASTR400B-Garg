# Cell 1

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

# Cell 2

# PURPOSE:
#   1) Define a function "Read" that reads our snapshot files from scratch
#   2) Demonstrate reading snapshot data for MW, M31, M33
#   3) Store or display relevant info

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
    label, value = line1.split()
    time = float(value)  # Myr

    line2 = file.readline()
    label2, value2 = line2.split()
    total = int(value2)  # total number of particles

    file.readline()
    file.readline()

    ptype_list = []
    m_list = []
    x_list = []
    y_list = []
    z_list = []
    vx_list = []
    vy_list = []
    vz_list = []

    for _ in range(total):
        line = file.readline()
        if not line:
            # in case of truncated file
            break

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

mw_file  = "MW_000.txt"
m31_file = "M31_000.txt"
m33_file = "M33_000.txt"

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

# Cell 3

# Center-of-Mass and Orbit Computations
#
# 1) Takes in the snapshot data array (type, m, x, y, z, vx, vy, vz)
# 2) Filters on a chosen particle type (1=halo, 2=disk, 3=bulge)
# 3) Implements the iterative COM approach for position (COM_P)
# 4) Implements a method for COM velocity (COM_V)
# 
# Then we'll compute the COM for MW, M31, and M33 as a demonstration.

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

print("A new 'CenterOfMass' class has been defined from scratch.")

# Cell 4

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

# Cell 5
# uncomment only to run and save file when needed

#build COM catalogue for all snapshots

galaxy_folders = ["MW", "M31", "M33"]
ptype_list     = [1, 2, 3]          # 1 = Halo, 2 = Disk, 3 = Bulge
records        = []

# helper to squash NaN / Inf to 0.0  (avoids six identical if‑blocks)
_clean = lambda v: 0.0 if (np.isnan(v) or np.isinf(v)) else v

for gal in galaxy_folders:
    for filename in sorted(glob.glob(f"{gal}/{gal}_*.txt")):
        time_myr, total_p, data_array = Read(filename)
        snap_str = filename.split("_")[-1].replace(".txt", "")

        for ptype in ptype_list:
            com      = CenterOfMass(data_array, ptype)
            # COM position / velocity (fall back to zeros on failure)
            try:
                xcom, ycom, zcom = com.COM_P(delta=0.1)
                vxcom, vycom, vzcom = com.COM_V(xcom, ycom, zcom, rvmax=15.0)
            except Exception:
                xcom = ycom = zcom = vxcom = vycom = vzcom = 0.0

            # clean NaN / Inf once, with helper
            xcom, ycom, zcom = map(_clean, (xcom, ycom, zcom))
            vxcom, vycom, vzcom = map(_clean, (vxcom, vycom, vzcom))

            records.append(
                dict(galaxy=gal, snapshot=snap_str, time_Myr=time_myr,
                     ptype=ptype, xcom=xcom, ycom=ycom, zcom=zcom,
                     vxcom=vxcom, vycom=vycom, vzcom=vzcom)
            )

df = pd.DataFrame(records)
df.to_csv("F1_All_COM.csv", index=False)
print(f"Saved COM data to F1_All_COM.csv with {len(df)} rows.")

# Cell 6

# Define a MassProfile class
#
# PURPOSE:
# For a galaxy's snapshot data (mw_data, m31_data, or m33_data) plus that galaxy's center (xCOM,yCOM,zCOM),
# we can compute the enclosed mass at any given radius for each ptype or the total.

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
        self.G = 4.498768e-6

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

        r_array = np.atleast_1d(radius)

        Menc_list = []
        for rmax in r_array:
            inside_idx = np.where(rr < rmax)[0]
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

# Cell 7

def compute_jacobi_radius(r_m31_m33, M_m33, Menc_m31):
    """Jacobi radius: R_J = r * (M_M33 / (2 M_enc_M31))^(1/3)"""
    return 0.0 if Menc_m31 <= 0 else r_m31_m33 * (M_m33 / (2.0*Menc_m31))**(1/3)

def jacobi_usage():
    """
    1) Load COM data           3) build MassProfile for M31
    2) loop over snapshots     4) compute Jacobi radius
    5) save to F2_JacobiRadius.csv
    """
    com_df = pd.read_csv("F1_All_COM.csv")
    # ensure numeric snapshot column once
    if 'snap_int' not in com_df.columns:
        com_df['snap_int'] = com_df['snapshot'].astype(int)
    all_snaps = sorted(com_df['snap_int'].unique())

    def get_com_row(gal, snap, ptype):
        return com_df.query(
            "(galaxy == @gal) & (ptype == @ptype) & (snap_int == @snap)"
        ).squeeze()  

    jaco_records = []
    for snap in all_snaps:
        snap_str = f"{snap:03d}"
        m31_file = f"M31/M31_{snap_str}.txt"
        m33_file = f"M33/M33_{snap_str}.txt"
        if not (os.path.exists(m31_file) and os.path.exists(m33_file)):
            continue

        _, _, data_m31 = Read(m31_file)
        _, _, data_m33 = Read(m33_file)

        row_m31 = get_com_row("M31", snap, 2)
        row_m33 = get_com_row("M33", snap, 2)
        if row_m31.empty or row_m33.empty:
            continue

        # separation
        dx = row_m33.xcom - row_m31.xcom
        dy = row_m33.ycom - row_m31.ycom
        dz = row_m33.zcom - row_m31.zcom
        r_m31_m33 = np.sqrt(dx*dx + dy*dy + dz*dz)

        M31prof   = MassProfile(data_m31, row_m31.xcom, row_m31.ycom, row_m31.zcom)
        Menc_m31  = M31prof.MassEnclosedTotal(r_m31_m33)

        M33prof   = MassProfile(data_m33, row_m33.xcom, row_m33.ycom, row_m33.zcom)
        M33_total = M33prof.MassEnclosedTotal(300.0)

        RJ = compute_jacobi_radius(r_m31_m33, M33_total, Menc_m31)

        jaco_records.append(dict(snapshot=snap_str, snap_int=snap,
                                 time_Myr=row_m31.time_Myr,
                                 r_M31_M33=r_m31_m33, M31_enc=Menc_m31,
                                 M33_total=M33_total, JacobiR=RJ))

    pd.DataFrame(jaco_records).to_csv("F2_JacobiRadius.csv", index=False)
    print(f"Saved {len(jaco_records)} rows to F2_JacobiRadius.csv.")

# now run
jacobi_usage()
# Cell 8

df_jacobi = pd.read_csv("F2_JacobiRadius.csv")
plt.plot(df_jacobi["time_Myr"], df_jacobi["JacobiR"], '-')
plt.xlabel("Time (Myr)")
plt.ylabel("Jacobi Radius (kpc)")
plt.title("Jacobi Radius Over Time")
plt.savefig("Fig1 Jacobi Radius Over Time.png", dpi=300)

plt.show()

# Cell 9 ── M 33 stellar‑mass loss versus Jacobi radius
# (pandas, numpy already imported earlier)

def dist3d(x, y, z, x0, y0, z0):
    """
    Compute the 3D Euclidean distance from (x,y,z) to (x0,y0,z0).
    """
    return np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    
def mass_loss_m33():
    """
    For every snapshot:
        • read M33 file and its COM (ptype = 2)
        • take Jacobi radius R_J from F2_JacobiRadius.csv
        • sum stellar mass (ptype = 2 or 3) inside R_J
        • compare to initial stellar mass (snapshot 0) → bound fraction
    Saves results to F3_M33_MassLoss.csv
    """
    jaco_df = pd.read_csv("F2_JacobiRadius.csv")
    com_df  = pd.read_csv("F1_All_COM.csv")

    # ensure numeric snapshot column exists once
    if 'snap_int' not in jaco_df.columns:
        jaco_df['snap_int'] = jaco_df['snapshot'].astype(int)
    if 'snap_int' not in com_df.columns:
        com_df['snap_int']  = com_df['snapshot'].astype(int)

    m33_com_df = com_df.query("(galaxy == 'M33') & (ptype == 2)")

    # ---------- helpers ----------------------------------------------------
    def get_jacobi_and_com(snap_int):
        """Return (time, R_J, xCOM, yCOM, zCOM) for a given snapshot."""
        row_j = jaco_df.query("snap_int == @snap_int")
        row_c = m33_com_df.query("snap_int == @snap_int")
        if len(row_j) != 1 or len(row_c) != 1:
            return None
        return (row_j.time_Myr.values[0], row_j.JacobiR.values[0],
                row_c.xcom.values[0],   row_c.ycom.values[0],  row_c.zcom.values[0])
    # -----------------------------------------------------------------------

    # --- initial stellar mass (snapshot 0 or earliest available) -----------
    zero_snap = 0 if 0 in jaco_df.snap_int.values else jaco_df.snap_int.min()
    init_vals = get_jacobi_and_com(zero_snap)
    if init_vals is None:
        print("No valid snapshot for initial mass. Aborting.")
        return
    t0, RJ0, x0, y0, z0 = init_vals

    init_file = f"M33/M33_{zero_snap:03d}.txt"
    _, _, data0 = Read(init_file)

    star_idx0 = np.where((data0['type'] == 2) | (data0['type'] == 3))[0]
    # total stellar mass (no radius cut, same as original)
    M33_init_star = np.sum(data0['m'][star_idx0]) * 1e10  # Msun
    print(f"Initial M33 stellar mass (snapshot {zero_snap:03d}) = {M33_init_star:.2e} Msun")

    # ---------------- iterate over snapshots ------------------------------
    bound_records = []
    for snap in sorted(jaco_df.snap_int.unique()):
        vals = get_jacobi_and_com(snap)
        if vals is None:
            continue
        time_myr, RJ, xcom, ycom, zcom = vals

        m33_file = f"M33/M33_{snap:03d}.txt"
        if not os.path.exists(m33_file):
            continue

        _, _, data = Read(m33_file)
        star_idx = np.where((data['type'] == 2) | (data['type'] == 3))[0]

        # distance of stellar particles from COM
        rr = dist3d(data['x'][star_idx], data['y'][star_idx], data['z'][star_idx],
                    xcom, ycom, zcom)

        inside_idx = star_idx[np.where(rr < RJ)]
        Mstar_bound = np.sum(data['m'][inside_idx]) * 1e10  # Msun
        frac_bound  = Mstar_bound / M33_init_star

        bound_records.append(dict(snapshot=f"{snap:03d}", snap_int=snap,
                                  time_Myr=time_myr, JacobiR=RJ,
                                  Mstar_bound=Mstar_bound, frac_bound=frac_bound))

    pd.DataFrame(bound_records).to_csv("F3_M33_MassLoss.csv", index=False)
    print(f"Mass‑loss results saved to F3_M33_MassLoss.csv with {len(bound_records)} rows.")


# run the analysis
mass_loss_m33()

# Cell 10 ── plot M 33 mass‑loss history  (imports already present)

def plot_m33_mass_loss():
    df_loss = pd.read_csv("F3_M33_MassLoss.csv")
    if df_loss.empty:
        print("No data in F3_M33_MassLoss.csv. Exiting.")
        return

    initial_mass = df_loss.Mstar_bound.iloc[0] / df_loss.frac_bound.iloc[0]
    final_mass   = df_loss.Mstar_bound.iloc[-1]
    frac_lost    = 1.0 - df_loss.frac_bound.iloc[-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_loss.time_Myr, df_loss.frac_bound, '-', label="M33 Bound Fraction")

    legend_title = (f"Initial = {initial_mass:.2e} Msun\n"
                    f"Final   = {final_mass:.2e} Msun\n"
                    f"Lost    = {frac_lost*100:.1f}%")
    ax.legend(title=legend_title)

    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel("Fraction of M33 Stellar Mass Bound")
    ax.set_title("M33 Mass Loss Over Time")
    plt.savefig("Fig2 M33 Mass Loss Over Time.png", dpi=300)
    plt.show()


plot_m33_mass_loss()

# Cell 11

# ───────────────────────────────────────────────────────────────
# Face‑on transformation, surface‑density profile & fitting tools
# ───────────────────────────────────────────────────────────────

def rotate_to_face_on(positions, velocities):
    """
    Rotate a disk to face‑on by aligning the total angular‑momentum vector
    with the z‑axis.  Returns new_pos, new_vel (N, 3) arrays.
    """
    L = np.sum(np.cross(positions, velocities), axis=0)
    Lmag = np.linalg.norm(L)
    if Lmag == 0:
        return positions, velocities
    Lhat = L / Lmag

    zhat = np.array([0, 0, 1.0])
    v    = np.cross(Lhat, zhat)
    s    = np.linalg.norm(v)
    c    = np.dot(Lhat, zhat)
    if s == 0:                         # already aligned
        return positions, velocities

    vx  = np.array([[   0, -v[2],  v[1]],
                    [ v[2],    0, -v[0]],
                    [-v[1],  v[0],    0]])
    R   = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)

    return positions @ R.T, velocities @ R.T


def surface_density_profile(positions, nbins=50, rmax=None):
    """
    Number‑based radial surface‑density profile for a face‑on disk.
    Returns r_mid, sigma.
    """
    r = np.hypot(positions[:, 0], positions[:, 1])
    if rmax is None:
        rmax = np.percentile(r, 99.0)

    edges  = np.linspace(0, rmax, nbins + 1)
    r_mid  = 0.5 * (edges[1:] + edges[:-1])
    sigma  = np.zeros(nbins)

    for i in range(nbins):
        r_in, r_out = edges[i], edges[i + 1]
        area        = np.pi * (r_out**2 - r_in**2)
        sigma[i]    = np.count_nonzero((r >= r_in) & (r < r_out)) / area
    return r_mid, sigma


# --- fitting functions ----------------------------------------------------

def sersic_function(r, I0, re, n):
    b = 2.0 * n - 1.0 / 3.0                 # rough b(n) approximation
    return I0 * np.exp(-b * ((r / re)**(1 / n) - 1.0))


def exponential_function(r, I0, rd):
    return I0 * np.exp(-r / rd)


def fit_sersic(r, sigma, guess=(1.0, 1.0, 1.0), use_exponential=False):
    """
    Fit either a Sersic (default) or exponential (if use_exponential=True)
    to the (r, sigma) profile.  Returns popt, pcov from curve_fit.
    """
    mask = sigma > 0
    r_fit, y_fit = r[mask], sigma[mask]
    if r_fit.size < 3:
        return None, None
    try:
        if use_exponential:
            return curve_fit(exponential_function, r_fit, y_fit, p0=guess[:2])
        else:
            return curve_fit(sersic_function,      r_fit, y_fit, p0=guess)
    except Exception:                       # fit failed
        return None, None


# ───────────────────────────────────────────────────────────────
# Main analysis: M33 disk‑profile evolution
# ───────────────────────────────────────────────────────────────

def disk_profile():
    """
    Loop over all M33 snapshots, rotate the stellar disk face‑on, build a
    radial surface‑density profile, fit exponential & Sersic models, and
    save the fit parameters to *F4_M33_DiskProfileFits.csv*.
    """
    com_df  = pd.read_csv("F1_All_COM.csv")
    m33_df  = com_df.query("(galaxy == 'M33') & (ptype == 2)").copy()
    if 'snap_int' not in m33_df.columns:
        m33_df['snap_int'] = m33_df['snapshot'].astype(int)

    records = []
    for snap in sorted(m33_df.snap_int.unique()):
        row = m33_df.loc[m33_df.snap_int == snap].squeeze()
        if row.empty:
            continue

        xcom, ycom, zcom, time_myr = row[['xcom', 'ycom', 'zcom', 'time_Myr']]
        fname = f"M33/M33_{snap:03d}.txt"
        if not os.path.exists(fname):
            continue

        _, _, data = Read(fname)
        star_idx   = np.where((data['type'] == 2) | (data['type'] == 3))[0]

        pos = np.column_stack((data['x'][star_idx] - xcom,
                               data['y'][star_idx] - ycom,
                               data['z'][star_idx] - zcom))
        vel = np.column_stack((data['vx'][star_idx],
                               data['vy'][star_idx],
                               data['vz'][star_idx]))

        pos_face, _ = rotate_to_face_on(pos, vel)

        r_mid, sigma = surface_density_profile(pos_face, nbins=40, rmax=20.0)
        popt_exp, _  = fit_sersic(r_mid, sigma,
                                  guess=(sigma.max(), 2.0),
                                  use_exponential=True)
        popt_ser, _  = fit_sersic(r_mid, sigma,
                                  guess=(sigma.max(), 2.0, 1.0),
                                  use_exponential=False)

        records.append({
            "snapshot"   : snap,
            "time_Myr"   : time_myr,
            "exp_I0"     : popt_exp[0] if popt_exp is not None else 0.0,
            "exp_r_d"    : popt_exp[1] if popt_exp is not None else 0.0,
            "sersic_I0"  : popt_ser[0] if popt_ser is not None else 0.0,
            "sersic_re"  : popt_ser[1] if popt_ser is not None else 0.0,
            "sersic_n"   : popt_ser[2] if popt_ser is not None else 0.0
        })

    pd.DataFrame(records).to_csv("F4_M33_DiskProfileFits.csv", index=False)
    print(f"Created F4_M33_DiskProfileFits.csv with {len(records)} rows.")


# ---- optional execution ---------------------------------------------------
disk_profile()

# Cell 12

def compute_disk_kinematics(new_pos, new_vel, nbins=20, rmax=None):
    """
    Given face‑on positions/velocities, return σ_rad, σ_tan, σ_z in radial bins.
    """
    x, y, z = new_pos.T
    vx, vy, vz = new_vel.T

    r   = np.hypot(x, y)
    phi = np.arctan2(y, x)

    v_rad =  vx*np.cos(phi) + vy*np.sin(phi)
    v_tan = -vx*np.sin(phi) + vy*np.cos(phi)

    if rmax is None:
        rmax = np.percentile(r, 99.0)

    edges      = np.linspace(0, rmax, nbins + 1)
    r_mid      = 0.5 * (edges[:-1] + edges[1:])
    sigma_rad  = np.zeros(nbins)
    sigma_tan  = np.zeros(nbins)
    sigma_z    = np.zeros(nbins)

    for i, (rin, rout) in enumerate(zip(edges[:-1], edges[1:])):
        idx = (r >= rin) & (r < rout)
        if idx.sum() >= 2:                          # need ≥2 points for std
            sigma_rad[i] = np.std(v_rad[idx])
            sigma_tan[i] = np.std(v_tan[idx])
            sigma_z[i]   = np.std(vz[idx])

    return r_mid, sigma_rad, sigma_tan, sigma_z


def disk_kinematics():
    """
    Rotate each M33 snapshot face‑on, measure σ’s in radial bins,
    and save to *F5_M33_Kinematics.csv* in long format.
    """
    com_df  = pd.read_csv("F1_All_COM.csv")
    m33_df  = com_df.query("(galaxy == 'M33') & (ptype == 2)").copy()
    if 'snap_int' not in m33_df.columns:
        m33_df['snap_int'] = m33_df['snapshot'].astype(int)

    records = []
    for snap in sorted(m33_df.snap_int.unique()):
        row = m33_df.loc[m33_df.snap_int == snap].squeeze()
        if row.empty:
            continue

        xcom, ycom, zcom, time_myr = row[['xcom', 'ycom', 'zcom', 'time_Myr']]
        fname = f"M33/M33_{snap:03d}.txt"
        if not os.path.exists(fname):
            continue

        _, _, data = Read(fname)
        star_idx   = np.where((data['type'] == 2) | (data['type'] == 3))[0]

        pos = np.column_stack((data['x'][star_idx] - xcom,
                               data['y'][star_idx] - ycom,
                               data['z'][star_idx] - zcom))
        vel = np.column_stack((data['vx'][star_idx],
                               data['vy'][star_idx],
                               data['vz'][star_idx]))

        pos_face, vel_face = rotate_to_face_on(pos, vel)
        r_mid, s_r, s_t, s_z = compute_disk_kinematics(pos_face, vel_face,
                                                       nbins=20, rmax=20.0)

        # long‑format rows
        for i in range(len(r_mid)):
            records.append(dict(snapshot=snap, time_Myr=time_myr,
                                bin_index=i, r_mid=r_mid[i],
                                sigma_rad=s_r[i], sigma_tan=s_t[i], sigma_z=s_z[i]))

    pd.DataFrame(records).to_csv("F5_M33_Kinematics.csv", index=False)
    print(f"disk_kinematics_demo finished — wrote {len(records)} rows.")

# run command
disk_kinematics()

# Cell 14
# Dummy test to see if the code worked or not
df_kin = pd.read_csv("F5_M33_Kinematics.csv")
df_100 = df_kin[df_kin['snapshot']==100]
plt.plot(df_100['r_mid'], df_100['sigma_z'], 'o-')
plt.xlabel("Radius (kpc)")
plt.ylabel("Vertical Velocity Dispersion (km/s)")
plt.title("M33 Kinematics at snapshot=100")
plt.show()

# Cell 15

def final_reporting():
    """
    Creates final plots/tables for the paper:
      1) M33 mass fraction vs time (from F3_M33_MassLoss.csv)
      2) M33 disk profile fits (from F4_M33_DiskProfileFits.csv)
      3) M33 disk velocity dispersions, but one figure per snapshot (from F5_M33_Kinematics.csv)
      4) Summaries saved in a text/csv in 'figures/'.

    Adjust as needed for your final suite of results.
    """

    ########################################################################
    # 0) Make a 'figures' folder if not present
    ########################################################################
    if not os.path.exists("figures"):
        os.mkdir("figures")

    ########################################################################
    # 1) Plot M33 disk profile fits (F4_M33_DiskProfileFits.csv)
    ########################################################################
    diskfits_file = "F4_M33_DiskProfileFits.csv"
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
        plt.savefig("Fig3 M33_DiskProfileFits.png")
        plt.close(fig)
        print("Saved Fig3 M33_DiskProfileFits.png")

    ########################################################################
    # 2) Disk Kinematics: We handle F5_M33_Kinematics.csv
    #    But we produce a SEPARATE figure for each snapshot => no huge multi-subplot
    ########################################################################
    kin_file = "F5_M33_Kinematics.csv"
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
            #plt.savefig(outname)                        # Uncomment to save images
            plt.close(fig)
            #plt.show()                                  # Uncomment to save images
            #print(f"Saved {outname}")                   # Uncomment to save images

    ########################################################################
    # 3) Summarize final numeric results in a text or CSV
    ########################################################################
    massloss_file = "F3_M33_MassLoss.csv"
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
            "description":"F6_Final M33 sersic n",
            "snapshot": last_row_fit["snapshot"],
            "time_Myr": last_row_fit["time_Myr"],
            "value": last_row_fit["sersic_n"]
        })

    if len(summary_rows)>0:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv("F6_Final_Summary.csv", index=False)
        print("Wrote a small summary table to Final_Summary.csv.")
    else:
        print("No final summary to write (missing input files).")

    print("All final plotting/reporting steps completed.")

# Let's it
final_reporting()

# Cell 16  ── warp & scale‑height analysis for M 33
# (numpy, pandas, matplotlib already imported earlier)

# -------------------------------------------------------------------------
# Helper: rotate coordinates so the disk‑angular‑momentum vector is the z‑axis
# -------------------------------------------------------------------------
def RotateFrame(x, y, z, L):
    """
    Rotate coordinates so the angular‑momentum vector **L** aligns with +z.
    Returns x', y', z' arrays of same length.
    """
    Lhat = L / np.linalg.norm(L)
    zhat = np.array([0.0, 0.0, 1.0])
    v    = np.cross(Lhat, zhat)
    s    = np.linalg.norm(v)
    c    = np.dot(Lhat, zhat)
    if s == 0:                              # already aligned
        return x, y, z

    K  = np.array([[    0, -v[2],  v[1]],
                   [ v[2],     0, -v[0]],
                   [-v[1],  v[0],     0]])
    R  = np.eye(3) + K + K @ K * ((1 - c) / s**2)

    xyz_rot = R @ np.vstack((x, y, z))
    return xyz_rot[0], xyz_rot[1], xyz_rot[2]


# -------------------------------------------------------------------------
# Warp / scale‑height measurement helpers (unique to this cell)
# -------------------------------------------------------------------------
def compute_angular_momentum(x, y, z, vx, vy, vz, m):
    """Total angular‑momentum vector ∑ m (r × v)."""
    Lx = np.sum(m * (y * vz - z * vy))
    Ly = np.sum(m * (z * vx - x * vz))
    Lz = np.sum(m * (x * vy - y * vx))
    return np.array([Lx, Ly, Lz])


def angle_between_vectors(a, b):
    """Return the angle (rad) between two 3‑D vectors."""
    cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.arccos(cosang)


def measure_warp_and_scaleheight(x, y, z, vx, vy, vz, m, nbins=10):
    """
    In radial bins, compute the local tilt (warp) and vertical scale height.
    Returns bin_mid, warp_angle[deg], scale_height[kpc].
    """
    L_global = compute_angular_momentum(x, y, z, vx, vy, vz, m)
    x_r, y_r, z_r = RotateFrame(x, y, z, L_global)
    vx_r, vy_r, vz_r = RotateFrame(vx, vy, vz, L_global)

    r     = np.hypot(x_r, y_r)
    rmax  = r.max()
    edges = np.linspace(0, rmax, nbins + 1)
    mid   = 0.5 * (edges[:-1] + edges[1:])

    warp_angle   = np.full(nbins, np.nan)
    scale_height = np.full(nbins, np.nan)

    for i, (rin, rout) in enumerate(zip(edges[:-1], edges[1:])):
        idx = (r >= rin) & (r < rout)
        if idx.sum() < 10:
            continue

        L_local = compute_angular_momentum(x_r[idx], y_r[idx], z_r[idx],
                                           vx_r[idx], vy_r[idx], vz_r[idx],
                                           m[idx])
        warp_angle[i]   = np.degrees(angle_between_vectors(L_local, L_global))
        scale_height[i] = np.std(z_r[idx])          # simple σ_z proxy

    return mid, warp_angle, scale_height


def analyze_morphology_for_snapshots(snaprange=range(0, 802, 50),
                                     datapath='M33/',
                                     output_file='F7_M33_morphology_results.txt'):
    """
    Loop over snapshots, compute warp & scale‑height, and write to *output_file*.
    Relies on the existing CenterOfMass and Read() functions defined earlier.
    """
    from pathlib import Path
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w') as f:
        f.write("# snap  time[Myr]  R_bin(kpc)  warp[deg]  h_z[kpc]\n")

        for snap in snaprange:
            fname = Path(datapath) / f"M33_{snap:03d}.txt"
            if not fname.exists():
                continue

            # Use Read() to load the snapshot instead of np.loadtxt
            time_val, total_val, data = Read(str(fname))
            # Create a CenterOfMass object with the loaded data (ptype=2 for disk)
            com = CenterOfMass(data, ptype=2)

            # Use the iterative method to get the COM position:
            time = time_val  # or com.time if that's set internally
            xcom, ycom, zcom = com.COM_P(delta=0.1)
            vxcom, vycom, vzcom = com.COM_V(xcom, ycom, zcom, rvmax=15.0)

            # shift to COM frame
            x  = com.x - xcom
            y  = com.y - ycom
            z  = com.z - zcom
            vx = com.vx - vxcom
            vy = com.vy - vycom
            vz = com.vz - vzcom
            m  = com.m

            mid, warp, h_z = measure_warp_and_scaleheight(x, y, z, vx, vy, vz, m, nbins=10)

            for R, w, h in zip(mid, warp, h_z):
                f.write(f"{snap:03d}  {time:9.2f}  {R:6.2f}  {w:6.2f}  {h:7.3f}\n")

    print(f"Done. Results saved to {out_path}")

# execution:
analyze_morphology_for_snapshots(snaprange=range(0, 802, 1),
                                 datapath='M33/',
                                 output_file='F7_M33_morphology_results.txt')

# Cell 17‑18  ── morphology‑results visualisation
# (numpy, pandas, matplotlib already imported earlier)

RESULTS_FILE = "F7_M33_morphology_results.txt"   # change if stored elsewhere

# ------------------------------------------------------------------
# 1.  Load the results table once
# ------------------------------------------------------------------
df = (pd.read_csv(RESULTS_FILE,
                  comment="#",
                  delim_whitespace=True,
                  names=["snapshot", "time_Myr", "R_kpc",
                         "warp_deg", "scale_kpc"])
        .dropna(subset=["warp_deg", "scale_kpc"]))           # drop incomplete bins

unique_snaps = np.sort(df.snapshot.unique())

# ------------------------------------------------------------------
# 2.  Warp angle vs radius for each snapshot
# ------------------------------------------------------------------
plt.figure()
for snap in unique_snaps:
    sub = df[df.snapshot == snap]
    plt.plot(sub.R_kpc, sub.warp_deg, label=f"snap {snap}")
plt.xlabel("Radius (kpc)")
plt.ylabel("Warp angle (deg)")
plt.title("M33 Disk Warp vs Radius")
plt.tight_layout()
#plt.savefig("Fig4 warp_vs_radius.png", dpi=300)
plt.show()

# ------------------------------------------------------------------
# 3.  Scale height vs radius for each snapshot
# ------------------------------------------------------------------
plt.figure()
for snap in unique_snaps:
    sub = df[df.snapshot == snap]
    plt.plot(sub.R_kpc, sub.scale_kpc, label=f"snap {snap}")
plt.xlabel("Radius (kpc)")
plt.ylabel("Scale height (kpc)")
plt.title("M33 Disk Scale Height vs Radius")
plt.tight_layout()
#plt.savefig("Fig5 scaleheight_vs_radius.png", dpi=300)
plt.show()

# ------------------------------------------------------------------
# 4.  Snapshot‑averaged trends (mean over the 10 radial bins)
# ------------------------------------------------------------------
snap_stats = (df.groupby("snapshot")
                .agg(time_Myr=("time_Myr", "first"),
                     warp_mean=("warp_deg",  "mean"),
                     scale_mean=("scale_kpc", "mean"))
                .reset_index())

# 4a. Mean warp vs time
plt.figure()
plt.plot(snap_stats.time_Myr, snap_stats.warp_mean)
plt.xlabel("Time (Myr)")
plt.ylabel("Mean warp angle (deg)")
plt.title("Disk Warp – Snapshot‑Averaged Trend")
plt.tight_layout()
plt.savefig("Fig4 warp_vs_time_avg.png", dpi=300)
plt.show()

# 4b. Mean scale height vs time
plt.figure()
plt.plot(snap_stats.time_Myr, snap_stats.scale_mean)
plt.xlabel("Time (Myr)")
plt.ylabel("Mean scale height (kpc)")
plt.title("Scale Height – Snapshot‑Averaged Trend")
plt.tight_layout()
plt.savefig("Fig5 scaleheight_vs_time_avg.png", dpi=300)
plt.show()


import nbformat

notebook_path = 'Project Code V5.ipynb'

nb = nbformat.read(notebook_path, as_version=4)

total_code_lines = 0

for cell in nb.cells:
    if cell.cell_type == 'code':
        lines = cell.source.splitlines()
        total_code_lines += len(lines)

print(f"Total lines of code in the notebook: {total_code_lines}")


