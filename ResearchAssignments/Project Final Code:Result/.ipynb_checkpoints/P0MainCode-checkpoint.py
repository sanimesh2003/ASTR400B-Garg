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

from P1Read import Read

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

from P2CenterOfMass import CenterOfMass

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
df.to_csv("All_COM.csv", index=False)
print(f"Saved COM data to All_COM.csv with {len(df)} rows.")

# Cell 6
# Define a MassProfile class
#
# PURPOSE:
# For a galaxy's snapshot data (mw_data, m31_data, or m33_data) plus that galaxy's center (xCOM,yCOM,zCOM),
# we can compute the enclosed mass at any given radius for each ptype or the total.
from P3MassProfile import MassProfile

# Cell 7
# Jacobi Functions
from P4JacobiRadius import compute_jacobi_radius
from P4JacobiRadius import jacobi_usage

# now run
#jacobi_usage()

# Cell 8
df_jacobi = pd.read_csv("JacobiRadius.csv")
plt.plot(df_jacobi["time_Myr"], df_jacobi["JacobiR"], '-')
plt.xlabel("Time (Myr)")
plt.ylabel("Jacobi Radius (kpc)")
plt.title("Jacobi Radius Over Time")
plt.savefig("Jacobi Radius Over Time.png", dpi=300)
plt.show()

# Cell 9 ── M 33 stellar‑mass loss versus Jacobi radius
# (pandas, numpy already imported earlier)

from P5MassLoss import dist3d, mass_loss_m33

# run the analysis
mass_loss_m33()

# Cell 10 ── plot M 33 mass‑loss history  (imports already present)

def plot_m33_mass_loss():
    df_loss = pd.read_csv("M33_MassLoss.csv")
    if df_loss.empty:
        print("No data in M33_MassLoss.csv. Exiting.")
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
    plt.savefig("M33 Mass Loss Over Time.png", dpi=300)
    plt.show()


plot_m33_mass_loss()

from P6DiskProfiler import DiskProfiler

profiler = DiskProfiler()
profiler.run()
from P7DiskKinematics import compute_disk_kinematics,disk_kinematics

disk_kinematics()
# Cell 14

df_kin = pd.read_csv("M33_Kinematics.csv")
df_100 = df_kin[df_kin['snapshot']==100]
plt.plot(df_100['r_mid'], df_100['sigma_z'], 'o-')
plt.xlabel("Radius (kpc)")
plt.ylabel("Vertical Velocity Dispersion (km/s)")
plt.title("M33 Kinematics at snapshot=100")
plt.show()

from P8FinalReporting import final_reporting

final_reporting()
from P9MorphologyAnalyzer import MorphologyAnalyzer

analyzer = MorphologyAnalyzer(datapath='M33/', output_file='M33_morphology_results.txt')
analyzer.analyze_snapshots(snaprange=range(0, 802, 1))

# Cell 17‑18  ── morphology‑results visualisation

RESULTS_FILE = "M33_morphology_results.txt"

# 1. Load the results table once
df = (pd.read_csv(RESULTS_FILE,
                  comment="#",
                  delim_whitespace=True,
                  names=["snapshot", "time_Myr", "R_kpc",
                         "warp_deg", "scale_kpc"])
        .dropna(subset=["warp_deg", "scale_kpc"]))

unique_snaps = np.sort(df.snapshot.unique())

# 2. Warp angle vs radius for each snapshot
plt.figure()
for snap in unique_snaps:
    sub = df[df.snapshot == snap]
    plt.plot(sub.R_kpc, sub.warp_deg, label=f"snap {snap}")
plt.xlabel("Radius (kpc)")
plt.ylabel("Warp angle (deg)")
plt.title("M33 Disk Warp vs Radius")
plt.tight_layout()
plt.savefig("warp_vs_radius.png", dpi=300)
plt.show()

# 3. Scale height vs radius for each snapshot
plt.figure()
for snap in unique_snaps:
    sub = df[df.snapshot == snap]
    plt.plot(sub.R_kpc, sub.scale_kpc, label=f"snap {snap}")
plt.xlabel("Radius (kpc)")
plt.ylabel("Scale height (kpc)")
plt.title("M33 Disk Scale Height vs Radius")
plt.tight_layout()
plt.savefig("scaleheight_vs_radius.png", dpi=300)
plt.show()

# 4. Snapshot‑averaged trends (mean over the 10 radial bins)
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
plt.savefig("warp_vs_time_avg.png", dpi=300)
plt.show()

# 4b. Mean scale height vs time
plt.figure()
plt.plot(snap_stats.time_Myr, snap_stats.scale_mean)
plt.xlabel("Time (Myr)")
plt.ylabel("Mean scale height (kpc)")
plt.title("Scale Height – Snapshot‑Averaged Trend")
plt.tight_layout()
plt.savefig("scaleheight_vs_time_avg.png", dpi=300)
plt.show()

import nbformat

notebook_path = 'Project Code V6.ipynb'

nb = nbformat.read(notebook_path, as_version=4)

total_code_lines = 0

for cell in nb.cells:
    if cell.cell_type == 'code':
        lines = cell.source.splitlines()
        total_code_lines += len(lines)

print(f"Total lines of code in the notebook: {total_code_lines}")


