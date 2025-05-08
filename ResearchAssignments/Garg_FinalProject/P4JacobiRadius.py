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
from P3MassProfile import MassProfile

def compute_jacobi_radius(r_m31_m33, M_m33, Menc_m31):
    """Jacobi radius: R_J = r * (M_M33 / (2 M_enc_M31))^(1/3)"""
    return 0.0 if Menc_m31 <= 0 else r_m31_m33 * (M_m33 / (2.0*Menc_m31))**(1/3)

def jacobi_usage():
    """
    1) Load COM data           3) build MassProfile for M31
    2) loop over snapshots     4) compute Jacobi radius
    5) save to JacobiRadius.csv
    """
    com_df = pd.read_csv("All_COM.csv")
    # ensure numeric snapshot column once
    if 'snap_int' not in com_df.columns:
        com_df['snap_int'] = com_df['snapshot'].astype(int)
    all_snaps = sorted(com_df['snap_int'].unique())

    def get_com_row(gal, snap, ptype):
        return com_df.query(
            "(galaxy == @gal) & (ptype == @ptype) & (snap_int == @snap)"
        ).squeeze()  

    num_potential_records = len(all_snaps)
    snapshot_arr = np.empty(num_potential_records, dtype='<U3') # For strings like '000'
    snap_int_arr = np.zeros(num_potential_records, dtype=int)
    time_Myr_arr = np.zeros(num_potential_records, dtype=float)
    r_M31_M33_arr = np.zeros(num_potential_records, dtype=float)
    M31_enc_arr = np.zeros(num_potential_records, dtype=float)
    M33_total_arr = np.zeros(num_potential_records, dtype=float)
    JacobiR_arr = np.zeros(num_potential_records, dtype=float)
    
    record_idx = 0

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
        
        # Check if rows are empty or if time_Myr is missing (can happen if squeeze() returns Series)
        if row_m31.empty or row_m33.empty or not hasattr(row_m31, 'time_Myr'):
            continue

        # separation
        dx = row_m33.xcom - row_m31.xcom
        dy = row_m33.ycom - row_m31.ycom
        dz = row_m33.zcom - row_m31.zcom
        r_m31_m33 = np.sqrt(dx*dx + dy*dy + dz*dz)

        if r_m31_m33 == 0: # Avoid issues if distance is zero
            continue

        M31prof   = MassProfile(data_m31, row_m31.xcom, row_m31.ycom, row_m31.zcom)
        Menc_m31  = M31prof.MassEnclosedTotal(r_m31_m33)

        M33prof   = MassProfile(data_m33, row_m33.xcom, row_m33.ycom, row_m33.zcom)
        M33_total = M33prof.MassEnclosedTotal(300.0) # Using 300kpc as in original context

        RJ = compute_jacobi_radius(r_m31_m33, M33_total, Menc_m31)

        snapshot_arr[record_idx] = snap_str
        snap_int_arr[record_idx] = snap
        time_Myr_arr[record_idx] = row_m31.time_Myr
        r_M31_M33_arr[record_idx] = r_m31_m33
        M31_enc_arr[record_idx] = Menc_m31
        M33_total_arr[record_idx] = M33_total
        JacobiR_arr[record_idx] = RJ
        record_idx += 1

    data_for_df = {
        'snapshot': snapshot_arr[:record_idx],
        'snap_int': snap_int_arr[:record_idx],
        'time_Myr': time_Myr_arr[:record_idx],
        'r_M31_M33': r_M31_M33_arr[:record_idx],
        'M31_enc': M31_enc_arr[:record_idx],
        'M33_total': M33_total_arr[:record_idx],
        'JacobiR': JacobiR_arr[:record_idx]
    }
    
    jaco_df = pd.DataFrame(data_for_df)
    jaco_df.to_csv("JacobiRadius.csv", index=False)
    print(f"Saved {record_idx} rows to JacobiRadius.csv.")
    
