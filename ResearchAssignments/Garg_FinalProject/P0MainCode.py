# Authors of this code : Aniemsh Garg, Google Gemini
#
# Aniemsh Garg  - code structures and writing comments
# Google Gemini - used as a debugging tool,
#                 also used to create better commenting structure
#
# Labs and homeworks from the class ASTR400B taught by Dr Gurtina Besla,
# were a resource in sourcing and completion of my code

# # P11: M33 Mass Loss Analysis Notebook
#
# This notebook calculates the mass loss of M33 within its Jacobi Radius
# during the simulated MW-M31-M33 interaction, using data from
# van der Marel et al. (2012) simulation snapshots.
# It integrates logic from the provided P1-P5 Python scripts.
# It generates Figure 3 (Context: Separation & Jacobi Radius) and
# Figure 4 (Result: Mass Loss Fraction) for the ASTR 400B report.

# %%
# --- Cell 1: Imports & Setup ---

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import os
import glob
from pathlib import Path # Using pathlib for cleaner path handling

# Import necessary functions/classes from provided modules
from P1Read import Read
from P2CenterOfMass import CenterOfMass
from P3MassProfile import MassProfile
from P4JacobiRadius import compute_jacobi_radius # Just need the formula function
from P5MassLoss import dist3d # Need the distance function

print("Imports successful.")

# Define base directory for simulation data
# Assumes M31, M33, MW folders are subdirectories of this path
# MODIFY THIS PATH AS NEEDED
BASE_DATA_DIR = Path("./") # Assumes data folders are in the same dir as notebook

# Define output filenames
COM_FILENAME = "All_COM.csv"
JACOBI_FILENAME = "JacobiRadius.csv"
MASSLOSS_FILENAME = "M33_MassLoss.csv"
FIG3_FILENAME = "Fig3_Separation_Jacobi.png"
FIG4_FILENAME = "Fig4_M33_MassLoss.png"

# Define particle types (consistent with provided code)
PTYPE_HALO = 1
PTYPE_DISK = 2
PTYPE_BULGE = 3
PTYPE_STARS = [PTYPE_DISK, PTYPE_BULGE] # Combine disk & bulge for stellar mass

# Global plotting style (optional, can customize)
plt.rcParams.update({'font.size': 14,
                     'axes.labelsize': 14,
                     'axes.titlesize': 16,
                     'legend.fontsize': 12,
                     'xtick.labelsize': 12,
                     'ytick.labelsize': 12,
                     'figure.figsize': (8, 6)})

print("Setup complete.")
# --- Cell 2: Generate Center of Mass Data ---

print(f"Checking for {COM_FILENAME}...")
if os.path.exists(COM_FILENAME):
    print(f"{COM_FILENAME} found. Loading existing data.")
    com_df = pd.read_csv(COM_FILENAME)
    # Ensure numeric snapshot column exists
    if 'snap_int' not in com_df.columns:
         com_df['snap_int'] = com_df['snapshot'].astype(int)
    print(f"Loaded {len(com_df)} rows.")
else:
    print(f"{COM_FILENAME} not found. Computing COM data...")
    galaxy_folders = ["MW", "M31", "M33"]
    ptype_list     = [PTYPE_HALO, PTYPE_DISK, PTYPE_BULGE]
    records        = []
    snap_files_found = 0

    # Helper to clean NaN/Inf
    _clean = lambda v: 0.0 if (np.isnan(v) or np.isinf(v)) else v

    for gal in galaxy_folders:
        data_path = BASE_DATA_DIR / gal
        glob_pattern = str(data_path / f"{gal}_*.txt") # Ensure correct path joining
        print(f"  Processing files in: {data_path} (using pattern: {glob_pattern})")
        
        snapshot_files = sorted(glob.glob(glob_pattern))
        if not snapshot_files:
             print(f"  Warning: No snapshot files found for {gal} matching pattern {glob_pattern}")
             continue
             
        snap_files_found += len(snapshot_files)

        for filename in snapshot_files:
            try:
                time_myr, total_p, data_array = Read(filename)
                # Extract snapshot number robustly
                snap_str = Path(filename).stem.split('_')[-1]
                snap_int = int(snap_str) # Store integer version too

                print(f"    Processing {gal} snapshot {snap_str} (Time: {time_myr:.1f} Myr)")

                for ptype in ptype_list:
                    # Skip bulge if galaxy is M33 (assuming no bulge like in some models)
                    if gal == "M33" and ptype == PTYPE_BULGE:
                        continue
                        
                    com = CenterOfMass(data_array, ptype)
                    
                    # Check if particles of this type exist
                    if len(com.m) == 0:
                        # print(f"      No particles of type {ptype} found for {gal} snap {snap_str}.")
                        xcom, ycom, zcom = 0.0, 0.0, 0.0
                        vxcom, vycom, vzcom = 0.0, 0.0, 0.0
                    else:
                         # Calculate COM position / velocity
                        try:
                            # Use a smaller tolerance for potentially faster convergence
                            xcom, ycom, zcom = com.COM_P(delta=0.05)
                            # Use a reasonable radius for velocity calculation
                            vxcom, vycom, vzcom = com.COM_V(xcom, ycom, zcom, rvmax=15.0)
                        except Exception as e:
                            print(f"      Error calculating COM for {gal} snap {snap_str} ptype {ptype}: {e}")
                            xcom = ycom = zcom = vxcom = vycom = vzcom = 0.0

                    # Clean NaN / Inf
                    xcom, ycom, zcom = map(_clean, (xcom, ycom, zcom))
                    vxcom, vycom, vzcom = map(_clean, (vxcom, vycom, vzcom))

                    records.append(
                        dict(galaxy=gal, snapshot=snap_str, snap_int=snap_int, time_Myr=time_myr,
                             ptype=ptype, xcom=xcom, ycom=ycom, zcom=zcom,
                             vxcom=vxcom, vycom=vzcom) # Small typo fix: vzcom
                    )
            except Exception as e:
                 print(f"  Error processing file {filename}: {e}")
                 continue # Skip to next file

    if not records:
         raise RuntimeError("No COM records were generated. Check data paths and file contents.")
         
    com_df = pd.DataFrame(records)
    com_df.to_csv(COM_FILENAME, index=False)
    print(f"Saved COM data to {COM_FILENAME} with {len(com_df)} rows from {snap_files_found} snapshot files.")

print("--- Cell 2 (COM Data) Finished ---")

# --- Cell 3: Calculate Jacobi Radii & Separation ---

print(f"Checking for {JACOBI_FILENAME}...")
if os.path.exists(JACOBI_FILENAME):
    print(f"{JACOBI_FILENAME} found. Loading existing data.")
    jaco_df = pd.read_csv(JACOBI_FILENAME)
     # Ensure numeric snapshot column exists
    if 'snap_int' not in jaco_df.columns:
        jaco_df['snap_int'] = jaco_df['snapshot'].astype(int)
    print(f"Loaded {len(jaco_df)} rows.")
    # Store unique snaps found in the CSV for later use
    processed_snaps = sorted(jaco_df['snap_int'].unique())

else:
    print(f"{JACOBI_FILENAME} not found. Calculating Jacobi Radii...")
    # Ensure COM data is loaded
    if 'com_df' not in locals():
        if os.path.exists(COM_FILENAME):
            print(f"Loading {COM_FILENAME} for Jacobi calculation...")
            com_df = pd.read_csv(COM_FILENAME)
            if 'snap_int' not in com_df.columns:
                com_df['snap_int'] = com_df['snapshot'].astype(int)
        else:
            raise FileNotFoundError(f"{COM_FILENAME} needed but not found or generated.")

    # Find common snapshots present for M31 and M33 disk (ptype=2)
    m31_snaps = set(com_df.query("galaxy == 'M31' and ptype == @PTYPE_DISK")['snap_int'])
    m33_snaps = set(com_df.query("galaxy == 'M33' and ptype == @PTYPE_DISK")['snap_int'])
    common_snaps = sorted(list(m31_snaps.intersection(m33_snaps)))
    
    if not common_snaps:
         raise ValueError("No common snapshots found for M31 and M33 disk particles in COM data.")

    # Helper to get a specific row from the COM DataFrame
    def get_com_row(gal, snap, ptype):
        # Use snap_int for querying
        query_str = f"(galaxy == '{gal}') & (ptype == {ptype}) & (snap_int == {snap})"
        result = com_df.query(query_str)
        if result.empty:
            # print(f"Warning: No COM data for {gal} snap {snap} ptype {ptype}")
            return None # Return None if no row found
        # If multiple rows match (shouldn't happen with snap_int), take the first
        return result.iloc[0]


    jaco_records = []
    processed_snaps = [] # Keep track of snaps we actually process

    for snap_int in common_snaps:
        snap_str = f"{snap_int:03d}"
        m31_file = BASE_DATA_DIR / f"M31/M31_{snap_str}.txt"
        m33_file = BASE_DATA_DIR / f"M33/M33_{snap_str}.txt"

        print(f"  Processing Jacobi Radius for snapshot {snap_str}...")

        if not (m31_file.exists() and m33_file.exists()):
            print(f"    Skipping snap {snap_str}: Missing data file(s).")
            continue

        # Read data
        try:
            time_m31, _, data_m31 = Read(str(m31_file))
            time_m33, _, data_m33 = Read(str(m33_file))
            # Use time from one, assuming they match for the same snapshot
            time_myr = time_m31
        except Exception as e:
             print(f"    Skipping snap {snap_str}: Error reading data file - {e}")
             continue

        # Get COM rows for disk particles (ptype=2 used for position)
        row_m31 = get_com_row("M31", snap_int, PTYPE_DISK)
        row_m33 = get_com_row("M33", snap_int, PTYPE_DISK)

        if row_m31 is None or row_m33 is None:
            print(f"    Skipping snap {snap_str}: Missing COM data.")
            continue

        # --- Calculate Separation R ---
        dx = row_m33.xcom - row_m31.xcom
        dy = row_m33.ycom - row_m31.ycom
        dz = row_m33.zcom - row_m31.zcom
        r_m31_m33 = np.sqrt(dx*dx + dy*dy + dz*dz)

        if r_m31_m33 == 0: # Avoid division by zero later
             print(f"    Skipping snap {snap_str}: Zero distance between M31 and M33.")
             continue

        # --- Calculate M_host(R) (M31 mass within R) ---
        try:
            M31_prof = MassProfile(data_m31, row_m31.xcom, row_m31.ycom, row_m31.zcom)
            # Use MassEnclosedTotal which sums ptypes 1, 2, 3
            Menc_m31 = M31_prof.MassEnclosedTotal(r_m31_m33)
        except Exception as e:
             print(f"    Skipping snap {snap_str}: Error calculating M31 enclosed mass - {e}")
             continue

        # --- Calculate M_sat (Total M33 Mass - Approximation) ---
        # Using total mass within a large radius (e.g., 300 kpc) as done in P4
        # This avoids the complexity of iteration for M_sat(R_J)
        try:
            M33_prof = MassProfile(data_m33, row_m33.xcom, row_m33.ycom, row_m33.zcom)
            # Use MassEnclosedTotal which sums ptypes 1, 2, 3
            # Using 300 kpc as the large radius like in P4JacobiRadius.py
            M33_total_approx = M33_prof.MassEnclosedTotal(300.0)
        except Exception as e:
             print(f"    Skipping snap {snap_str}: Error calculating M33 total mass - {e}")
             continue
             
        if Menc_m31 <= 0 or M33_total_approx <= 0:
             print(f"    Skipping snap {snap_str}: Non-positive mass calculated (M31_enc={Menc_m31}, M33_tot={M33_total_approx}).")
             continue

        # --- Calculate Jacobi Radius R_J ---
        RJ = compute_jacobi_radius(r_m31_m33, M33_total_approx, Menc_m31)

        jaco_records.append(dict(snapshot=snap_str, snap_int=snap_int,
                                 time_Myr=time_myr,
                                 r_M31_M33=r_m31_m33, M31_enc=Menc_m31,
                                 M33_total=M33_total_approx, JacobiR=RJ))
        processed_snaps.append(snap_int)
        
    if not jaco_records:
         raise RuntimeError("No Jacobi Radius records were generated. Check input data and COM file.")

    jaco_df = pd.DataFrame(jaco_records)
    jaco_df.to_csv(JACOBI_FILENAME, index=False)
    print(f"Saved Jacobi Radius data to {JACOBI_FILENAME} with {len(jaco_df)} rows.")
    processed_snaps = sorted(list(set(processed_snaps))) # Ensure sorted unique list

print("--- Cell 3 (Jacobi Radii) Finished ---")

# --- Cell 4: Calculate M33 Mass Loss ---

print(f"Checking for {MASSLOSS_FILENAME}...")
if os.path.exists(MASSLOSS_FILENAME):
    print(f"{MASSLOSS_FILENAME} found. Loading existing data.")
    mass_loss_df = pd.read_csv(MASSLOSS_FILENAME)
    if 'snap_int' not in mass_loss_df.columns:
         mass_loss_df['snap_int'] = mass_loss_df['snapshot'].astype(int)
    print(f"Loaded {len(mass_loss_df)} rows.")
else:
    print(f"{MASSLOSS_FILENAME} not found. Calculating mass loss...")
    # Ensure COM and Jacobi data are loaded/available
    if 'com_df' not in locals():
        if os.path.exists(COM_FILENAME):
            print(f"Loading {COM_FILENAME} for Mass Loss calculation...")
            com_df = pd.read_csv(COM_FILENAME)
            if 'snap_int' not in com_df.columns:
                com_df['snap_int'] = com_df['snapshot'].astype(int)
        else:
            raise FileNotFoundError(f"{COM_FILENAME} needed but not found or generated.")
    if 'jaco_df' not in locals():
        if os.path.exists(JACOBI_FILENAME):
            print(f"Loading {JACOBI_FILENAME} for Mass Loss calculation...")
            jaco_df = pd.read_csv(JACOBI_FILENAME)
            if 'snap_int' not in jaco_df.columns:
                 jaco_df['snap_int'] = jaco_df['snapshot'].astype(int)
            processed_snaps = sorted(jaco_df['snap_int'].unique()) # Use snaps from Jacobi file
        else:
             raise FileNotFoundError(f"{JACOBI_FILENAME} needed but not found or generated.")
             
    # Filter COM data for M33 disk (ptype=2)
    m33_com_df = com_df.query("(galaxy == 'M33') & (ptype == @PTYPE_DISK)").copy()
    if m33_com_df.empty:
         raise ValueError("No M33 disk COM data found.")

    # --- Determine Initial M33 Stellar Mass ---
    # Use the earliest snapshot successfully processed in the Jacobi step
    if not processed_snaps:
         raise ValueError("Cannot determine initial mass, no snapshots were processed for Jacobi radius.")
         
    zero_snap = min(processed_snaps)
    init_file = BASE_DATA_DIR / f"M33/M33_{zero_snap:03d}.txt"
    print(f"Calculating initial M33 stellar mass from snapshot {zero_snap:03d}...")

    if not init_file.exists():
        raise FileNotFoundError(f"Initial snapshot file {init_file} not found.")

    try:
        _, _, data0 = Read(str(init_file))
        # Select stars (disk + bulge types)
        star_idx0 = np.isin(data0['type'], PTYPE_STARS)
        # Sum mass of all stars (no radius cut initially)
        # Assuming mass in data file is in 1e10 Msun
        M33_init_star = np.sum(data0['m'][star_idx0]) * 1e10  # Total Msun
        if M33_init_star <= 0:
             raise ValueError(f"Calculated initial stellar mass is non-positive ({M33_init_star:.2e} Msun). Check data file {init_file}.")
        print(f"Initial M33 stellar mass (snapshot {zero_snap:03d}) = {M33_init_star:.2e} Msun")
    except Exception as e:
         raise RuntimeError(f"Error calculating initial M33 stellar mass: {e}")


    # --- Iterate Over Snapshots to Calculate Bound Mass ---
    bound_records = []
    for snap_int in processed_snaps: # Loop only over snaps where RJ was calculated
        snap_str = f"{snap_int:03d}"
        m33_file = BASE_DATA_DIR / f"M33/M33_{snap_str}.txt"
        print(f"  Processing Mass Loss for snapshot {snap_str}...")

        if not m33_file.exists():
            print(f"    Skipping snap {snap_str}: Missing data file.")
            continue

        # Get Jacobi Radius and M33 COM for this snapshot
        jaco_row = jaco_df.query("snap_int == @snap_int")
        com_row = m33_com_df.query("snap_int == @snap_int") # Use snap_int

        if jaco_row.empty or com_row.empty:
            print(f"    Skipping snap {snap_str}: Missing Jacobi or COM data.")
            continue

        # Extract values safely using .iloc[0] after ensuring not empty
        time_myr = jaco_row['time_Myr'].iloc[0]
        RJ = jaco_row['JacobiR'].iloc[0]
        xcom = com_row['xcom'].iloc[0]
        ycom = com_row['ycom'].iloc[0]
        zcom = com_row['zcom'].iloc[0]

        if RJ <= 0:
             print(f"    Skipping snap {snap_str}: Invalid Jacobi Radius ({RJ:.2f}).")
             continue

        # Read M33 data for this snapshot
        try:
            _, _, data = Read(str(m33_file))
            # Select M33 stars (types 2 or 3)
            star_idx = np.isin(data['type'], PTYPE_STARS)
            
            if not np.any(star_idx): # Check if any stars exist
                 print(f"    Skipping snap {snap_str}: No stellar particles found.")
                 Mstar_bound = 0.0
            else:
                # Calculate distances of stars from M33 COM
                stars_x = data['x'][star_idx]
                stars_y = data['y'][star_idx]
                stars_z = data['z'][star_idx]
                rr = dist3d(stars_x, stars_y, stars_z, xcom, ycom, zcom)

                # Find stars within Jacobi Radius
                inside_mask = rr < RJ
                
                # Check if any stars are inside RJ
                if not np.any(inside_mask):
                     #print(f"    Snap {snap_str}: No M33 stars found within R_J = {RJ:.2f} kpc.")
                     Mstar_bound = 0.0
                else:
                     # Sum mass of bound stars (use the original indices via inside_mask)
                     mass_arr = data['m'][star_idx] # Get masses of all stars first
                     Mstar_bound = np.sum(mass_arr[inside_mask]) * 1e10  # Total Msun
                     
        except Exception as e:
             print(f"    Skipping snap {snap_str}: Error processing M33 data - {e}")
             continue


        # Calculate fraction bound
        frac_bound = Mstar_bound / M33_init_star if M33_init_star > 0 else 0

        bound_records.append(dict(snapshot=snap_str, snap_int=snap_int,
                                  time_Myr=time_myr, JacobiR=RJ,
                                  Mstar_bound=Mstar_bound, frac_bound=frac_bound))

    if not bound_records:
        raise RuntimeError("No Mass Loss records were generated. Check input data.")
        
    mass_loss_df = pd.DataFrame(bound_records)
    mass_loss_df.to_csv(MASSLOSS_FILENAME, index=False)
    print(f"Mass loss results saved to {MASSLOSS_FILENAME} with {len(mass_loss_df)} rows.")

print("--- Cell 4 (Mass Loss) Finished ---")
# --- Cell 5: Generate Figure 3 (Context Plot - Separation & Jacobi Radius vs Time) ---

print("Generating Figure 3: Separation and Jacobi Radius vs Time...")
if 'jaco_df' not in locals():
    if os.path.exists(JACOBI_FILENAME):
        print(f"Loading {JACOBI_FILENAME} for plotting...")
        jaco_df = pd.read_csv(JACOBI_FILENAME)
    else:
        raise FileNotFoundError(f"{JACOBI_FILENAME} not found. Cannot generate Figure 3.")

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Separation Distance on primary y-axis
color = 'tab:blue'
ax1.set_xlabel('Time (Myr)')
ax1.set_ylabel('M31-M33 Separation (kpc)', color=color)
ax1.plot(jaco_df['time_Myr'], jaco_df['r_M31_M33'], color=color, linestyle='-', markersize=4, label='Separation (R)')
ax1.tick_params(axis='y', labelcolor=color)

# Create secondary y-axis for Jacobi Radius
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Jacobi Radius (kpc)', color=color)
ax2.plot(jaco_df['time_Myr'], jaco_df['JacobiR'], color=color, linestyle='--', markersize=4, label='Jacobi Radius (R$_J$)')
ax2.tick_params(axis='y', labelcolor=color)

# Add grid and legend
ax1.grid(True, linestyle=':', alpha=0.7)
fig.suptitle('M31-M33 Separation and M33 Jacobi Radius Over Time')
# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to prevent title overlap
plt.savefig(FIG3_FILENAME, dpi=300)
print(f"Saved context plot to {FIG3_FILENAME}")
plt.show()

print("--- Cell 5 (Figure 3) Finished ---")
# --- Cell 6: Generate Figure 4 (Result Plot - Mass Loss Fraction vs Time) ---

# Assuming necessary imports like matplotlib.pyplot as plt, pandas as pd, etc.
# Assuming MASSLOSS_FILENAME, BASE_DATA_DIR, Read, PTYPE_STARS, FIG4_FILENAME are defined earlier

print("Generating Figure 4: M33 Bound Stellar Mass Fraction vs Time...")
if 'mass_loss_df' not in locals():
    if os.path.exists(MASSLOSS_FILENAME):
        print(f"Loading {MASSLOSS_FILENAME} for plotting...")
        mass_loss_df = pd.read_csv(MASSLOSS_FILENAME)
    else:
        # Make sure FIG4_FILENAME is defined even if we can't plot, to avoid errors later
        if 'FIG4_FILENAME' not in locals(): FIG4_FILENAME = "Fig4_MassLoss_Error.png"
        raise FileNotFoundError(f"{MASSLOSS_FILENAME} not found. Cannot generate Figure 4.")
else:
    # Ensure mass_loss_df is not None if it exists
    if mass_loss_df is None:
       if 'FIG4_FILENAME' not in locals(): FIG4_FILENAME = "Fig4_MassLoss_Error.png"
       raise ValueError("mass_loss_df exists but is None. Cannot generate Figure 4.")


# Find initial and final fractions for annotation
if not mass_loss_df.empty:
    initial_frac = mass_loss_df['frac_bound'].iloc[0]
    final_frac = mass_loss_df['frac_bound'].iloc[-1]
    # Ensure initial_frac is not zero before calculating percentage loss this way
    percent_lost = (initial_frac - final_frac) * 100 if initial_frac != 0 else 0

    # Get initial mass used for normalization (calculated in Cell 4)
    # Added error handling for M33_init_star calculation
    M33_init_star_val = np.nan # Default to NaN
    if 'M33_init_star' in locals() and M33_init_star is not None:
         M33_init_star_val = M33_init_star
    else:
        try:
            zero_snap = min(mass_loss_df['snap_int']) # Use earliest snap from mass loss file
            # Check if BASE_DATA_DIR and Read are available
            if 'BASE_DATA_DIR' in locals() and 'Read' in locals():
                 init_file = BASE_DATA_DIR / f"M33/M33_{zero_snap:03d}.txt"
                 if init_file.exists():
                      _, _, data0 = Read(str(init_file))
                      # Check if PTYPE_STARS is defined
                      if 'PTYPE_STARS' in locals():
                           star_idx0 = np.isin(data0['type'], PTYPE_STARS)
                           M33_init_star_val = np.sum(data0['m'][star_idx0]) * 1e10
                      else:
                           print("Warning: PTYPE_STARS not defined, cannot calculate initial mass precisely.")
                 else:
                      print(f"Warning: Initial snapshot file {init_file} not found.")
            else:
                 print("Warning: BASE_DATA_DIR or Read function not available, cannot calculate initial mass.")
        except Exception as e:
            print(f"Warning: Error calculating initial mass: {e}")

    # Format annotation text carefully, handling potential NaN
    init_mass_str = f"{M33_init_star_val:.2e}" if not np.isnan(M33_init_star_val) else "N/A"
    annotation_text = (f"Initial Mass Ref: {init_mass_str} $M_\odot$\n"
                       f"Initial Bound Frac: {initial_frac:.3f}\n"
                       f"Final Bound Frac: {final_frac:.3f}\n"
                       f"Total Mass Lost: {percent_lost:.1f}%")
else:
    annotation_text = "No mass loss data found."
    final_frac = 0 # Default for ylim
    # Ensure FIG4_FILENAME is defined if needed for saving placeholder
    if 'FIG4_FILENAME' not in locals(): FIG4_FILENAME = "Fig4_MassLoss_NoData.png"


# --- Set Font Sizes ---
# Increase the base font size for most elements (labels, ticks, default text)
# Adjust this value (e.g., 14, 16, 18) as needed
BASE_FONT_SIZE_FIG4 = 14
plt.rcParams.update({'font.size': BASE_FONT_SIZE_FIG4})
# You can make titles/labels even larger relative to the base if desired
plt.rcParams['axes.titlesize'] = BASE_FONT_SIZE_FIG4 + 2 # e.g., 16
plt.rcParams['axes.labelsize'] = BASE_FONT_SIZE_FIG4 + 1 # e.g., 15
plt.rcParams['xtick.labelsize'] = BASE_FONT_SIZE_FIG4 # e.g., 14
plt.rcParams['ytick.labelsize'] = BASE_FONT_SIZE_FIG4 # e.g., 14
plt.rcParams['legend.fontsize'] = BASE_FONT_SIZE_FIG4 # e.g., 14 (if legend were used)


# --- Create Plot ---
fig, ax = plt.subplots(figsize=(10, 6)) # Consider adjusting figsize if needed

if not mass_loss_df.empty:
    # *** Increased markersize for visual balance ***
    ax.plot(mass_loss_df['time_Myr'], mass_loss_df['frac_bound'], linestyle='-', markersize=6, label='Bound Fraction') # Added marker='o'
else:
    # Use the base font size for this message too
    ax.text(0.5, 0.5, "No data to plot", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=BASE_FONT_SIZE_FIG4)

# Font sizes for labels and title are now controlled by rcParams above
ax.set_xlabel('Time (Myr)')
ax.set_ylabel('Fraction of Initial M33 Stellar Mass Bound within $R_J$')
ax.set_title('M33 Stellar Mass Loss Over Time')
ax.grid(True, linestyle=':', alpha=0.7)
ax.set_ylim(bottom=max(0, final_frac - 0.1), top=1.05) # Adjust ylim dynamically

# Add annotation box
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# *** Increased annotation fontsize (e.g., slightly smaller than axis labels) ***
ax.text(0.95, 0.95, annotation_text, transform=ax.transAxes, fontsize=BASE_FONT_SIZE_FIG4 -1, # e.g., 13
        verticalalignment='top', horizontalalignment='right', bbox=props)

# ax.legend() # Legend might be redundant if only one line, size controlled by rcParams

plt.tight_layout()

# Ensure FIG4_FILENAME is defined before saving
if 'FIG4_FILENAME' not in locals():
    FIG4_FILENAME = "Fig4_MassLoss_Plot.png" # Provide a default name if missing
    print(f"Warning: FIG4_FILENAME was not defined. Saving plot to {FIG4_FILENAME}")

plt.savefig(FIG4_FILENAME, dpi=300)
print(f"Saved result plot to {FIG4_FILENAME}")
plt.show()

# --- Restore default font sizes if necessary (optional) ---
# If you have more plots later in the script and don't want them to use these large sizes:
# plt.rcParams.update(plt.rcParamsDefault)

print("--- Cell 6 (Figure 4) Finished ---")


# --- Cell 7: Print Summary ---
# (The rest of your Cell 7 code follows)
# Note: Font settings for the plot do not affect the print statements in Cell 7.

print("Calculating final summary...")
# (Your existing Cell 7 code...)
if 'mass_loss_df' not in locals():
     if os.path.exists(MASSLOSS_FILENAME):
         mass_loss_df = pd.read_csv(MASSLOSS_FILENAME)
     else:
         mass_loss_df = pd.DataFrame() # Empty dataframe

if not mass_loss_df.empty:
     # Use the M33_init_star_val calculated earlier for consistency
     initial_mass_bound = mass_loss_df['Mstar_bound'].iloc[0]
     final_mass_bound = mass_loss_df['Mstar_bound'].iloc[-1]

     # This recalculation might be redundant if M33_init_star_val was correctly determined above
     if np.isnan(M33_init_star_val): # Attempt recalculation only if it failed earlier
         print("Attempting to recalculate initial mass for summary...")
         try:
            zero_snap = min(mass_loss_df['snap_int'])
            if 'BASE_DATA_DIR' in locals() and 'Read' in locals():
                init_file = BASE_DATA_DIR / f"M33/M33_{zero_snap:03d}.txt"
                if init_file.exists():
                     _, _, data0 = Read(str(init_file))
                     if 'PTYPE_STARS' in locals():
                          star_idx0 = np.isin(data0['type'], PTYPE_STARS)
                          M33_init_star_val = np.sum(data0['m'][star_idx0]) * 1e10
                     else: M33_init_star_val = np.nan
                else: M33_init_star_val = np.nan
            else: M33_init_star_val = np.nan
         except Exception as e:
            print(f"Recalculation error: {e}")
            M33_init_star_val = np.nan


     total_mass_lost_fraction = mass_loss_df['frac_bound'].iloc[0] - mass_loss_df['frac_bound'].iloc[-1]
     total_mass_lost_percent = total_mass_lost_fraction * 100

     init_mass_str_summary = f"{M33_init_star_val:.3e}" if not np.isnan(M33_init_star_val) else "N/A"

     print("\n--- Final Summary ---")
     print(f"Initial Total Stellar Mass Reference (Snap {min(mass_loss_df['snap_int'])}): {init_mass_str_summary} Msun")
     print(f"Initial Bound Stellar Mass (Snap {min(mass_loss_df['snap_int'])}): {initial_mass_bound:.3e} Msun (Fraction: {mass_loss_df['frac_bound'].iloc[0]:.3f})")
     print(f"Final Bound Stellar Mass (Snap {max(mass_loss_df['snap_int'])}):   {final_mass_bound:.3e} Msun (Fraction: {mass_loss_df['frac_bound'].iloc[-1]:.3f})")
     print(f"Total Fraction of Initial Mass Lost: {total_mass_lost_fraction:.3f} ({total_mass_lost_percent:.1f}%)")
     print("--------------------\n")
else:
     print("Could not generate summary, mass loss data not available.")

print("--- Cell 7 (Summary) Finished ---")

# %%
print("Notebook execution complete.")

