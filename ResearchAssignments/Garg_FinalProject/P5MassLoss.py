# Authors of this code : Aniemsh Garg, Google Gemini
#
# Aniemsh Garg  - code structures and writing comments
# Google Gemini - used as a debugging tool,
#                 also used to create better commenting structure
#
# Labs and homeworks from the class ASTR400B taught by Dr Gurtina Besla,
# were a resource in sourcing and completion of my code

# This code is new, not derived from the labs or homeworks used in class

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import os
import pandas as pd
import glob
from scipy.optimize import curve_fit
from P1Read import Read

def dist3d(x, y, z, x0, y0, z0):
    """
    Compute the 3D Euclidean distance from (x,y,z) to (x0,y0,z0).
    """
    return np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    
def mass_loss_m33():
    """
    For every snapshot:
        • read M33 file and its COM (ptype = 2)
        • take Jacobi radius R_J from JacobiRadius.csv
        • sum stellar mass (ptype = 2 or 3) inside R_J
        • compare to initial stellar mass (snapshot 0) → bound fraction
    Saves results to M33_MassLoss.csv
    """
    jaco_df = pd.read_csv("JacobiRadius.csv")
    com_df  = pd.read_csv("All_COM.csv")

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

    # --- initial stellar mass (snapshot 0 or earliest available) -----------
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
    print(f"Initial M33 stellar mass (snapshot {zero_snap:03d}) = {M33_init_star:.2e} Msun")

    # ---------------- iterate over snapshots ------------------------------
    unique_snaps = sorted(jaco_df.snap_int.unique())
    num_potential_records = len(unique_snaps)

    snapshot_arr = np.empty(num_potential_records, dtype='<U3') # For strings like '000'
    snap_int_arr = np.zeros(num_potential_records, dtype=int)
    time_Myr_arr = np.zeros(num_potential_records, dtype=float)
    JacobiR_arr = np.zeros(num_potential_records, dtype=float)
    Mstar_bound_arr = np.zeros(num_potential_records, dtype=float)
    frac_bound_arr = np.zeros(num_potential_records, dtype=float)
    
    record_idx = 0

    for snap in unique_snaps:
        vals = get_jacobi_and_com(snap)
        if vals is None:
            continue
        time_myr, RJ, xcom, ycom, zcom = vals

        m33_file = f"M33/M33_{snap:03d}.txt"
        if not os.path.exists(m33_file):
            continue

        _, _, data = Read(m33_file)
        star_idx = np.where((data['type'] == 2) | (data['type'] == 3))[0]

        if len(star_idx) == 0: # No stars of the specified types found
            Mstar_bound = 0.0
        else:
            # distance of stellar particles from COM
            rr = dist3d(data['x'][star_idx], data['y'][star_idx], data['z'][star_idx],
                        xcom, ycom, zcom)

            inside_idx = star_idx[np.where(rr < RJ)]
            Mstar_bound = np.sum(data['m'][inside_idx]) * 1e10  # Msun
        
        if M33_init_star == 0: # Avoid division by zero
            frac_bound = 0.0
        else:
            frac_bound  = Mstar_bound / M33_init_star

        snapshot_arr[record_idx] = f"{snap:03d}"
        snap_int_arr[record_idx] = snap
        time_Myr_arr[record_idx] = time_myr
        JacobiR_arr[record_idx] = RJ
        Mstar_bound_arr[record_idx] = Mstar_bound
        frac_bound_arr[record_idx] = frac_bound
        record_idx += 1

    data_for_df = {
        'snapshot': snapshot_arr[:record_idx],
        'snap_int': snap_int_arr[:record_idx],
        'time_Myr': time_Myr_arr[:record_idx],
        'JacobiR': JacobiR_arr[:record_idx],
        'Mstar_bound': Mstar_bound_arr[:record_idx],
        'frac_bound': frac_bound_arr[:record_idx]
    }

    bound_df = pd.DataFrame(data_for_df)
    bound_df.to_csv("M33_MassLoss.csv", index=False)
    print(f"Mass-loss results saved to M33_MassLoss.csv with {record_idx} rows.")
    
