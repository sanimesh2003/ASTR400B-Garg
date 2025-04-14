import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from P1Read import Read
from P6DiskProfiler import DiskProfiler
rotate_to_face_on = DiskProfiler.rotate_to_face_on

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
    and save to *M33_Kinematics.csv* in long format.
    """
    com_df  = pd.read_csv("All_COM.csv")
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

    pd.DataFrame(records).to_csv("M33_Kinematics.csv", index=False)
    print(f"disk_kinematics_demo finished — wrote {len(records)} rows.")
