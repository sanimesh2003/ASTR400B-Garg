import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import os
import pandas as pd
import glob
from scipy.optimize import curve_fit
from P1Read import Read
from P6DiskProfiler import DiskProfiler

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
            #plt.savefig(outname)                        # Uncomment to save images
            plt.close(fig)
            #plt.show()                                  # Uncomment to save images
            #print(f"Saved {outname}")                   # Uncomment to save images

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
