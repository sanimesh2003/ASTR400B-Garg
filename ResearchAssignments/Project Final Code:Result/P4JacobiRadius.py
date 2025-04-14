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
    """Jacobi radius: R_J = r * (M_M33 / (2 M_enc_M31))^(1/3)"""
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

    pd.DataFrame(jaco_records).to_csv("JacobiRadius.csv", index=False)
    print(f"Saved {len(jaco_records)} rows to JacobiRadius.csv.")
