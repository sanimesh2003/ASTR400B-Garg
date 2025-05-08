# Authors of this code : Aniemsh Garg, Google Gemini
#
# Aniemsh Garg  - code structures and writing comments
# Google Gemini - used as a debugging tool,
#                 also used to create better commenting structure
#
# Labs and homeworks from the class ASTR400B taught by Dr Gurtina Besla,
# were a resource in sourcing and completion of my code

# # P0: M33 Mass Loss Analysis Notebook
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
    
    # --- Determine total number of records to pre-allocate for COM data ---
    file_processing_tuples = []
    for gal_scan in galaxy_folders:
        data_path_scan = BASE_DATA_DIR / gal_scan
        glob_pattern_scan = str(data_path_scan / f"{gal_scan}_*.txt") # Ensure correct path joining
        current_snapshot_files = sorted(glob.glob(glob_pattern_scan))
        if not current_snapshot_files:
             print(f"  Warning: No snapshot files found for {gal_scan} matching pattern {glob_pattern_scan} (during pre-scan)")
        for f_scan in current_snapshot_files:
            file_processing_tuples.append({'gal': gal_scan, 'filename': f_scan, 'path': data_path_scan})

    max_com_records = len(file_processing_tuples) * len(ptype_list)
    if max_com_records == 0:
        print("Warning: No snapshot files found across all galaxy folders during pre-scan. COM data generation will be skipped or result in empty file.")
    
    # Pre-allocate arrays for COM data
    com_galaxy_arr = np.empty(max_com_records, dtype='<U3')  # For 'MW', 'M31', 'M33'
    com_snapshot_arr = np.empty(max_com_records, dtype='<U10') # For snapshot strings like '000'
    com_snap_int_arr = np.zeros(max_com_records, dtype=int)
    com_time_Myr_arr = np.zeros(max_com_records, dtype=float)
    com_ptype_arr = np.zeros(max_com_records, dtype=int)
    com_xcom_arr = np.zeros(max_com_records, dtype=float)
    com_ycom_arr = np.zeros(max_com_records, dtype=float)
    com_zcom_arr = np.zeros(max_com_records, dtype=float)
    com_vxcom_arr = np.zeros(max_com_records, dtype=float)
    com_vycom_arr = np.zeros(max_com_records, dtype=float)
    com_vzcom_arr = np.zeros(max_com_records, dtype=float)
    
    com_record_idx = 0
    snap_files_processed_count = 0 # To match original snap_files_found logic

    # Helper to clean NaN/Inf
    _clean = lambda v: 0.0 if (np.isnan(v) or np.isinf(v)) else v

    # Group files by galaxy for printing purposes, similar to original structure
    files_by_galaxy = {}
    for item in file_processing_tuples:
        if item['gal'] not in files_by_galaxy:
            files_by_galaxy[item['gal']] = []
        files_by_galaxy[item['gal']].append(item['filename'])

    for gal in galaxy_folders: # Iterate in defined order for consistent printing
        data_path = BASE_DATA_DIR / gal
        # Get files for this galaxy from our pre-scanned list
        snapshot_files_for_gal = files_by_galaxy.get(gal, [])

        if not snapshot_files_for_gal: # This check might be redundant if pre-scan warning was sufficient
            # This print was in the original code if glob returned empty for a gal
            # print(f"  Warning: No snapshot files found for {gal} matching pattern {str(data_path / f'{gal}_*.txt')}")
            continue # Skip if no files for this galaxy from pre-scan

        print(f"  Processing files in: {data_path} (using pattern: {str(data_path / f'{gal}_*.txt')})")
        snap_files_processed_count += len(snapshot_files_for_gal)

        for filename in snapshot_files_for_gal:
            try:
                time_myr, total_p, data_array = Read(filename)
                snap_str = Path(filename).stem.split('_')[-1]
                snap_int = int(snap_str)

                print(f"    Processing {gal} snapshot {snap_str} (Time: {time_myr:.1f} Myr)")

                for ptype in ptype_list:
                    if gal == "M33" and ptype == PTYPE_BULGE:
                        continue
                        
                    com = CenterOfMass(data_array, ptype)
                    
                    xcom_val, ycom_val, zcom_val = 0.0, 0.0, 0.0
                    vxcom_val, vycom_val, vzcom_val = 0.0, 0.0, 0.0

                    if len(com.m) == 0:
                        # print(f"      No particles of type {ptype} found for {gal} snap {snap_str}.")
                        pass # Values already initialized to 0.0
                    else:
                        try:
                            xcom_val, ycom_val, zcom_val = com.COM_P(delta=0.05)
                            vxcom_val, vycom_val, vzcom_val = com.COM_V(xcom_val, ycom_val, zcom_val, rvmax=15.0)
                        except Exception as e:
                            print(f"      Error calculating COM for {gal} snap {snap_str} ptype {ptype}: {e}")
                            # Values remain 0.0

                    xcom_val, ycom_val, zcom_val = map(_clean, (xcom_val, ycom_val, zcom_val))
                    vxcom_val, vycom_val, vzcom_val = map(_clean, (vxcom_val, vycom_val, vzcom_val))

                    # Assign to pre-allocated arrays
                    if com_record_idx < max_com_records:
                        com_galaxy_arr[com_record_idx] = gal
                        com_snapshot_arr[com_record_idx] = snap_str
                        com_snap_int_arr[com_record_idx] = snap_int
                        com_time_Myr_arr[com_record_idx] = time_myr
                        com_ptype_arr[com_record_idx] = ptype
                        com_xcom_arr[com_record_idx] = xcom_val
                        com_ycom_arr[com_record_idx] = ycom_val
                        com_zcom_arr[com_record_idx] = zcom_val
                        com_vxcom_arr[com_record_idx] = vxcom_val
                        # Original dict had `vycom=vzcom_val`. Assuming user wants correct assignment.
                        # If strict bug replication: com_vycom_arr[com_record_idx] = vzcom_val and no vzcom_arr.
                        # Given the comment "# Small typo fix: vzcom", interpreting as intent to fix/use all components.
                        com_vycom_arr[com_record_idx] = vycom_val 
                        com_vzcom_arr[com_record_idx] = vzcom_val
                        com_record_idx += 1
                    else:
                        print(f"Error: Exceeded pre-allocated array size for COM data. Max: {max_com_records}")
                        # This indicates an issue with pre-allocation logic or unexpected data volume.
                        break # from ptype loop
            except Exception as e:
                 print(f"  Error processing file {filename}: {e}")
                 if com_record_idx >= max_com_records: break # from filename loop if array full
                 continue 
            if com_record_idx >= max_com_records: break # from filename loop if array full
        if com_record_idx >= max_com_records: break # from gal loop if array full

    com_data_dict = {
        'galaxy': com_galaxy_arr[:com_record_idx],
        'snapshot': com_snapshot_arr[:com_record_idx],
        'snap_int': com_snap_int_arr[:com_record_idx],
        'time_Myr': com_time_Myr_arr[:com_record_idx],
        'ptype': com_ptype_arr[:com_record_idx],
        'xcom': com_xcom_arr[:com_record_idx],
        'ycom': com_ycom_arr[:com_record_idx],
        'zcom': com_zcom_arr[:com_record_idx],
        'vxcom': com_vxcom_arr[:com_record_idx],
        'vycom': com_vycom_arr[:com_record_idx],
        'vzcom': com_vzcom_arr[:com_record_idx]
    }
    com_df = pd.DataFrame(com_data_dict)

    if com_df.empty and len(file_processing_tuples) > 0 : # Files existed but no records made
         raise RuntimeError("No COM records were generated despite finding snapshot files. Check data paths and file contents.")
    elif com_df.empty and len(file_processing_tuples) == 0: # No files found at all
         raise RuntimeError("No COM records were generated. No snapshot files found.")
         
    com_df.to_csv(COM_FILENAME, index=False)
    print(f"Saved COM data to {COM_FILENAME} with {len(com_df)} rows from {snap_files_processed_count} snapshot files.")

print("--- Cell 2 (COM Data) Finished ---")

# --- Cell 3: Calculate Jacobi Radii & Separation ---

print(f"Checking for {JACOBI_FILENAME}...")
if os.path.exists(JACOBI_FILENAME):
    print(f"{JACOBI_FILENAME} found. Loading existing data.")
    jaco_df = pd.read_csv(JACOBI_FILENAME)
    if 'snap_int' not in jaco_df.columns:
        jaco_df['snap_int'] = jaco_df['snapshot'].astype(int)
    print(f"Loaded {len(jaco_df)} rows.")
    processed_snaps_np_arr = np.array(sorted(jaco_df['snap_int'].unique()), dtype=int)

else:
    print(f"{JACOBI_FILENAME} not found. Calculating Jacobi Radii...")
    if 'com_df' not in locals():
        if os.path.exists(COM_FILENAME):
            print(f"Loading {COM_FILENAME} for Jacobi calculation...")
            com_df = pd.read_csv(COM_FILENAME)
            if 'snap_int' not in com_df.columns:
                com_df['snap_int'] = com_df['snapshot'].astype(int)
        else:
            raise FileNotFoundError(f"{COM_FILENAME} needed but not found or generated.")

    m31_snaps = set(com_df.query("galaxy == 'M31' and ptype == @PTYPE_DISK")['snap_int'])
    m33_snaps = set(com_df.query("galaxy == 'M33' and ptype == @PTYPE_DISK")['snap_int'])
    common_snaps_list = sorted(list(m31_snaps.intersection(m33_snaps)))
    
    if not common_snaps_list:
         raise ValueError("No common snapshots found for M31 and M33 disk particles in COM data.")

    def get_com_row(gal, snap, ptype):
        query_str = f"(galaxy == '{gal}') & (ptype == {ptype}) & (snap_int == {snap})"
        result = com_df.query(query_str)
        if result.empty:
            return None
        return result.iloc[0]

    max_jaco_records = len(common_snaps_list)
    jaco_snapshot_arr = np.empty(max_jaco_records, dtype='<U10')
    jaco_snap_int_arr = np.zeros(max_jaco_records, dtype=int)
    jaco_time_Myr_arr = np.zeros(max_jaco_records, dtype=float)
    jaco_r_M31_M33_arr = np.zeros(max_jaco_records, dtype=float)
    jaco_M31_enc_arr = np.zeros(max_jaco_records, dtype=float)
    jaco_M33_total_arr = np.zeros(max_jaco_records, dtype=float)
    jaco_JacobiR_arr = np.zeros(max_jaco_records, dtype=float)
    
    processed_snaps_temp_arr = np.zeros(max_jaco_records, dtype=int) # For building processed_snaps
    jaco_record_idx = 0

    for snap_int in common_snaps_list:
        snap_str = f"{snap_int:03d}"
        m31_file = BASE_DATA_DIR / f"M31/M31_{snap_str}.txt"
        m33_file = BASE_DATA_DIR / f"M33/M33_{snap_str}.txt"

        print(f"  Processing Jacobi Radius for snapshot {snap_str}...")

        if not (m31_file.exists() and m33_file.exists()):
            print(f"    Skipping snap {snap_str}: Missing data file(s).")
            continue
        try:
            time_m31, _, data_m31 = Read(str(m31_file))
            time_m33, _, data_m33 = Read(str(m33_file))
            time_myr = time_m31
        except Exception as e:
             print(f"    Skipping snap {snap_str}: Error reading data file - {e}")
             continue

        row_m31 = get_com_row("M31", snap_int, PTYPE_DISK)
        row_m33 = get_com_row("M33", snap_int, PTYPE_DISK)

        if row_m31 is None or row_m33 is None:
            print(f"    Skipping snap {snap_str}: Missing COM data.")
            continue

        dx = row_m33.xcom - row_m31.xcom
        dy = row_m33.ycom - row_m31.ycom
        dz = row_m33.zcom - row_m31.zcom
        r_m31_m33 = np.sqrt(dx*dx + dy*dy + dz*dz)

        if r_m31_m33 == 0:
             print(f"    Skipping snap {snap_str}: Zero distance between M31 and M33.")
             continue
        try:
            M31_prof = MassProfile(data_m31, row_m31.xcom, row_m31.ycom, row_m31.zcom)
            Menc_m31 = M31_prof.MassEnclosedTotal(r_m31_m33)
        except Exception as e:
             print(f"    Skipping snap {snap_str}: Error calculating M31 enclosed mass - {e}")
             continue
        try:
            M33_prof = MassProfile(data_m33, row_m33.xcom, row_m33.ycom, row_m33.zcom)
            M33_total_approx = M33_prof.MassEnclosedTotal(300.0)
        except Exception as e:
             print(f"    Skipping snap {snap_str}: Error calculating M33 total mass - {e}")
             continue
             
        if Menc_m31 <= 0 or M33_total_approx <= 0:
             print(f"    Skipping snap {snap_str}: Non-positive mass calculated (M31_enc={Menc_m31}, M33_tot={M33_total_approx}).")
             continue

        RJ = compute_jacobi_radius(r_m31_m33, M33_total_approx, Menc_m31)

        if jaco_record_idx < max_jaco_records:
            jaco_snapshot_arr[jaco_record_idx] = snap_str
            jaco_snap_int_arr[jaco_record_idx] = snap_int
            jaco_time_Myr_arr[jaco_record_idx] = time_myr
            jaco_r_M31_M33_arr[jaco_record_idx] = r_m31_m33
            jaco_M31_enc_arr[jaco_record_idx] = Menc_m31
            jaco_M33_total_arr[jaco_record_idx] = M33_total_approx
            jaco_JacobiR_arr[jaco_record_idx] = RJ
            processed_snaps_temp_arr[jaco_record_idx] = snap_int # Store for unique processing
            jaco_record_idx += 1
        else:
            print(f"Error: Exceeded pre-allocated array size for Jacobi records. Max: {max_jaco_records}")
            break
        
    if jaco_record_idx == 0 and len(common_snaps_list) > 0:
         raise RuntimeError("No Jacobi Radius records were generated despite having common snapshots. Check input data and COM file.")
    elif len(common_snaps_list) == 0 : # Should have been caught by earlier check
         raise RuntimeError("No Jacobi Radius records generated as no common snapshots were found.")


    jaco_data_dict = {
        'snapshot': jaco_snapshot_arr[:jaco_record_idx],
        'snap_int': jaco_snap_int_arr[:jaco_record_idx],
        'time_Myr': jaco_time_Myr_arr[:jaco_record_idx],
        'r_M31_M33': jaco_r_M31_M33_arr[:jaco_record_idx],
        'M31_enc': jaco_M31_enc_arr[:jaco_record_idx],
        'M33_total': jaco_M33_total_arr[:jaco_record_idx],
        'JacobiR': jaco_JacobiR_arr[:jaco_record_idx]
    }
    jaco_df = pd.DataFrame(jaco_data_dict)
    jaco_df.to_csv(JACOBI_FILENAME, index=False)
    print(f"Saved Jacobi Radius data to {JACOBI_FILENAME} with {len(jaco_df)} rows.")
    # Ensure sorted unique list for processed_snaps_np_arr
    if jaco_record_idx > 0:
        processed_snaps_np_arr = np.unique(processed_snaps_temp_arr[:jaco_record_idx])
    else: # No records processed
        processed_snaps_np_arr = np.array([], dtype=int)


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
    if 'com_df' not in locals():
        if os.path.exists(COM_FILENAME):
            print(f"Loading {COM_FILENAME} for Mass Loss calculation...")
            com_df = pd.read_csv(COM_FILENAME)
            if 'snap_int' not in com_df.columns:
                com_df['snap_int'] = com_df['snapshot'].astype(int)
        else:
            raise FileNotFoundError(f"{COM_FILENAME} needed but not found or generated.")
    if 'jaco_df' not in locals(): # Should be populated if previous cell ran
        if os.path.exists(JACOBI_FILENAME):
            print(f"Loading {JACOBI_FILENAME} for Mass Loss calculation...")
            jaco_df = pd.read_csv(JACOBI_FILENAME)
            if 'snap_int' not in jaco_df.columns:
                 jaco_df['snap_int'] = jaco_df['snapshot'].astype(int)
            # processed_snaps_np_arr should be defined from cell 3 logic if jaco_df was created there
            if 'processed_snaps_np_arr' not in locals(): # Fallback if jaco_df loaded but processed_snaps_np_arr not set
                 processed_snaps_np_arr = np.array(sorted(jaco_df['snap_int'].unique()), dtype=int)
        else:
             raise FileNotFoundError(f"{JACOBI_FILENAME} needed but not found or generated.")
             
    m33_com_df = com_df.query("(galaxy == 'M33') & (ptype == @PTYPE_DISK)").copy()
    if m33_com_df.empty:
         raise ValueError("No M33 disk COM data found.")

    if not processed_snaps_np_arr.size: # Check if the numpy array is empty
         raise ValueError("Cannot determine initial mass, no snapshots were processed for Jacobi radius (processed_snaps_np_arr is empty).")
         
    zero_snap = np.min(processed_snaps_np_arr)
    init_file = BASE_DATA_DIR / f"M33/M33_{zero_snap:03d}.txt"
    print(f"Calculating initial M33 stellar mass from snapshot {zero_snap:03d}...")

    if not init_file.exists():
        raise FileNotFoundError(f"Initial snapshot file {init_file} not found.")

    M33_init_star = 0.0 # Initialize
    try:
        _, _, data0 = Read(str(init_file))
        star_idx0 = np.isin(data0['type'], PTYPE_STARS)
        M33_init_star = np.sum(data0['m'][star_idx0]) * 1e10
        if M33_init_star <= 0:
             raise ValueError(f"Calculated initial stellar mass is non-positive ({M33_init_star:.2e} Msun). Check data file {init_file}.")
        print(f"Initial M33 stellar mass (snapshot {zero_snap:03d}) = {M33_init_star:.2e} Msun")
    except Exception as e:
         raise RuntimeError(f"Error calculating initial M33 stellar mass: {e}")

    max_bound_records = len(processed_snaps_np_arr)
    bound_snapshot_arr = np.empty(max_bound_records, dtype='<U10')
    bound_snap_int_arr = np.zeros(max_bound_records, dtype=int)
    bound_time_Myr_arr = np.zeros(max_bound_records, dtype=float)
    bound_JacobiR_arr = np.zeros(max_bound_records, dtype=float)
    bound_Mstar_bound_arr = np.zeros(max_bound_records, dtype=float)
    bound_frac_bound_arr = np.zeros(max_bound_records, dtype=float)
    bound_record_idx = 0

    for snap_int_iter in processed_snaps_np_arr:
        snap_str_iter = f"{snap_int_iter:03d}"
        m33_file_iter = BASE_DATA_DIR / f"M33/M33_{snap_str_iter}.txt"
        print(f"  Processing Mass Loss for snapshot {snap_str_iter}...")

        if not m33_file_iter.exists():
            print(f"    Skipping snap {snap_str_iter}: Missing data file.")
            continue

        jaco_row = jaco_df.query("snap_int == @snap_int_iter")
        com_row = m33_com_df.query("snap_int == @snap_int_iter")

        if jaco_row.empty or com_row.empty:
            print(f"    Skipping snap {snap_str_iter}: Missing Jacobi or COM data.")
            continue

        time_myr_iter = jaco_row['time_Myr'].iloc[0]
        RJ_iter = jaco_row['JacobiR'].iloc[0]
        xcom_iter = com_row['xcom'].iloc[0]
        ycom_iter = com_row['ycom'].iloc[0]
        zcom_iter = com_row['zcom'].iloc[0]

        if RJ_iter <= 0:
             print(f"    Skipping snap {snap_str_iter}: Invalid Jacobi Radius ({RJ_iter:.2f}).")
             continue
        
        Mstar_bound_val = 0.0
        try:
            _, _, data_iter = Read(str(m33_file_iter))
            star_idx_iter = np.isin(data_iter['type'], PTYPE_STARS)
            
            if not np.any(star_idx_iter):
                 print(f"    Skipping snap {snap_str_iter}: No stellar particles found.")
                 # Mstar_bound_val remains 0.0
            else:
                stars_x = data_iter['x'][star_idx_iter]
                stars_y = data_iter['y'][star_idx_iter]
                stars_z = data_iter['z'][star_idx_iter]
                rr_iter = dist3d(stars_x, stars_y, stars_z, xcom_iter, ycom_iter, zcom_iter)
                inside_mask_iter = rr_iter < RJ_iter
                
                if not np.any(inside_mask_iter):
                     # Mstar_bound_val remains 0.0
                     pass
                else:
                     mass_arr_iter = data_iter['m'][star_idx_iter]
                     Mstar_bound_val = np.sum(mass_arr_iter[inside_mask_iter]) * 1e10
                     
        except Exception as e:
             print(f"    Skipping snap {snap_str_iter}: Error processing M33 data - {e}")
             continue

        frac_bound_val = Mstar_bound_val / M33_init_star if M33_init_star > 0 else 0.0
        
        if bound_record_idx < max_bound_records:
            bound_snapshot_arr[bound_record_idx] = snap_str_iter
            bound_snap_int_arr[bound_record_idx] = snap_int_iter
            bound_time_Myr_arr[bound_record_idx] = time_myr_iter
            bound_JacobiR_arr[bound_record_idx] = RJ_iter
            bound_Mstar_bound_arr[bound_record_idx] = Mstar_bound_val
            bound_frac_bound_arr[bound_record_idx] = frac_bound_val
            bound_record_idx += 1
        else:
            print(f"Error: Exceeded pre-allocated array size for Mass Loss records. Max: {max_bound_records}")
            break


    if bound_record_idx == 0 and len(processed_snaps_np_arr) > 0 :
        raise RuntimeError("No Mass Loss records were generated despite having processed snapshots. Check input data.")
    elif len(processed_snaps_np_arr) == 0:
         raise RuntimeError("No Mass Loss records were generated as there are no processed snapshots.")
        
    mass_loss_data_dict = {
        'snapshot': bound_snapshot_arr[:bound_record_idx],
        'snap_int': bound_snap_int_arr[:bound_record_idx],
        'time_Myr': bound_time_Myr_arr[:bound_record_idx],
        'JacobiR': bound_JacobiR_arr[:bound_record_idx],
        'Mstar_bound': bound_Mstar_bound_arr[:bound_record_idx],
        'frac_bound': bound_frac_bound_arr[:bound_record_idx]
    }
    mass_loss_df = pd.DataFrame(mass_loss_data_dict)
    mass_loss_df.to_csv(MASSLOSS_FILENAME, index=False)
    print(f"Mass loss results saved to {MASSLOSS_FILENAME} with {len(mass_loss_df)} rows.")

print("--- Cell 4 (Mass Loss) Finished ---")
# --- Cell 5: Generate Figure 3 (Context Plot - Separation & Jacobi Radius vs Time) ---

print("Generating Figure 3: Separation and Jacobi Radius vs Time...")
if 'jaco_df' not in locals(): # Should be defined if Cell 3 ran or loaded
    if os.path.exists(JACOBI_FILENAME):
        print(f"Loading {JACOBI_FILENAME} for plotting...")
        jaco_df = pd.read_csv(JACOBI_FILENAME)
    else:
        raise FileNotFoundError(f"{JACOBI_FILENAME} not found. Cannot generate Figure 3.")

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Time (Myr)')
ax1.set_ylabel('M31-M33 Separation (kpc)', color=color)
ax1.plot(jaco_df['time_Myr'], jaco_df['r_M31_M33'], color=color, linestyle='-', markersize=4, label='Separation (R)')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Jacobi Radius (kpc)', color=color)
ax2.plot(jaco_df['time_Myr'], jaco_df['JacobiR'], color=color, linestyle='--', markersize=4, label='Jacobi Radius (R$_J$)')
ax2.tick_params(axis='y', labelcolor=color)

ax1.grid(True, linestyle=':', alpha=0.7)
fig.suptitle('M31-M33 Separation and M33 Jacobi Radius Over Time')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.savefig(FIG3_FILENAME, dpi=300)
print(f"Saved context plot to {FIG3_FILENAME}")
plt.show()

print("--- Cell 5 (Figure 3) Finished ---")
# --- Cell 6: Generate Figure 4 (Result Plot - Mass Loss Fraction vs Time) ---

# Assuming necessary imports like matplotlib.pyplot as plt, pandas as pd, etc.
# Assuming MASSLOSS_FILENAME, BASE_DATA_DIR, Read, PTYPE_STARS, FIG4_FILENAME are defined earlier

print("Generating Figure 4: M33 Bound Stellar Mass Fraction vs Time...")
if 'mass_loss_df' not in locals(): # Should be defined if Cell 4 ran or loaded
    if os.path.exists(MASSLOSS_FILENAME):
        print(f"Loading {MASSLOSS_FILENAME} for plotting...")
        mass_loss_df = pd.read_csv(MASSLOSS_FILENAME)
    else:
        if 'FIG4_FILENAME' not in locals(): FIG4_FILENAME = "Fig4_MassLoss_Error.png"
        raise FileNotFoundError(f"{MASSLOSS_FILENAME} not found. Cannot generate Figure 4.")
else:
    if mass_loss_df is None: # Should not happen if logic is correct
       if 'FIG4_FILENAME' not in locals(): FIG4_FILENAME = "Fig4_MassLoss_Error.png"
       raise ValueError("mass_loss_df exists but is None. Cannot generate Figure 4.")


if not mass_loss_df.empty:
    initial_frac = mass_loss_df['frac_bound'].iloc[0]
    final_frac = mass_loss_df['frac_bound'].iloc[-1]
    percent_lost = (initial_frac - final_frac) * 100 if initial_frac != 0 else 0.0

    M33_init_star_val = np.nan 
    if 'M33_init_star' in locals() and M33_init_star is not None: # M33_init_star from Cell 4
         M33_init_star_val = M33_init_star
    else: # Attempt to recalculate if not available (e.g. if notebook cells run out of order)
        try:
            if not mass_loss_df.empty and 'snap_int' in mass_loss_df.columns:
                zero_snap_fig = min(mass_loss_df['snap_int']) 
                if 'BASE_DATA_DIR' in locals() and 'Read' in locals():
                     init_file_fig = BASE_DATA_DIR / f"M33/M33_{zero_snap_fig:03d}.txt"
                     if init_file_fig.exists():
                          _, _, data0_fig = Read(str(init_file_fig))
                          if 'PTYPE_STARS' in locals():
                               star_idx0_fig = np.isin(data0_fig['type'], PTYPE_STARS)
                               M33_init_star_val = np.sum(data0_fig['m'][star_idx0_fig]) * 1e10
                          else:
                               print("Warning: PTYPE_STARS not defined, cannot calculate initial mass precisely for figure.")
                     else:
                          print(f"Warning: Initial snapshot file {init_file_fig} not found for figure.")
                else:
                     print("Warning: BASE_DATA_DIR or Read function not available, cannot calculate initial mass for figure.")
            else:
                print("Warning: mass_loss_df is empty or missing 'snap_int', cannot determine initial mass for figure.")
        except Exception as e:
            print(f"Warning: Error calculating initial mass for figure: {e}")

    init_mass_str = f"{M33_init_star_val:.2e}" if not np.isnan(M33_init_star_val) else "N/A"
    annotation_text = (f"Initial Mass Ref: {init_mass_str} $M_\odot$\n"
                       f"Initial Bound Frac: {initial_frac:.3f}\n"
                       f"Final Bound Frac: {final_frac:.3f}\n"
                       f"Total Mass Lost: {percent_lost:.1f}%")
else:
    annotation_text = "No mass loss data found."
    final_frac = 0 
    if 'FIG4_FILENAME' not in locals(): FIG4_FILENAME = "Fig4_MassLoss_NoData.png"

BASE_FONT_SIZE_FIG4 = 14
plt.rcParams.update({'font.size': BASE_FONT_SIZE_FIG4})
plt.rcParams['axes.titlesize'] = BASE_FONT_SIZE_FIG4 + 2 
plt.rcParams['axes.labelsize'] = BASE_FONT_SIZE_FIG4 + 1 
plt.rcParams['xtick.labelsize'] = BASE_FONT_SIZE_FIG4 
plt.rcParams['ytick.labelsize'] = BASE_FONT_SIZE_FIG4 
plt.rcParams['legend.fontsize'] = BASE_FONT_SIZE_FIG4 

fig, ax = plt.subplots(figsize=(10, 6)) 

if not mass_loss_df.empty:
    ax.plot(mass_loss_df['time_Myr'], mass_loss_df['frac_bound'], linestyle='-', markersize=6, label='Bound Fraction') 
else:
    ax.text(0.5, 0.5, "No data to plot", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=BASE_FONT_SIZE_FIG4)

ax.set_xlabel('Time (Myr)')
ax.set_ylabel('Fraction of Initial M33 Stellar Mass Bound within $R_J$')
ax.set_title('M33 Stellar Mass Loss Over Time')
ax.grid(True, linestyle=':', alpha=0.7)
ax.set_ylim(bottom=max(0, final_frac - 0.1 if not mass_loss_df.empty else 0), top=1.05) 

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.95, 0.95, annotation_text, transform=ax.transAxes, fontsize=BASE_FONT_SIZE_FIG4 -1, 
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()

if 'FIG4_FILENAME' not in locals():
    FIG4_FILENAME = "Fig4_MassLoss_Plot.png" 
    print(f"Warning: FIG4_FILENAME was not defined. Saving plot to {FIG4_FILENAME}")

plt.savefig(FIG4_FILENAME, dpi=300)
print(f"Saved result plot to {FIG4_FILENAME}")
plt.show()

print("--- Cell 6 (Figure 4) Finished ---")

# --- Cell 7: Print Summary ---
print("Calculating final summary...")
if 'mass_loss_df' not in locals(): # Should be defined
     if os.path.exists(MASSLOSS_FILENAME):
         mass_loss_df = pd.read_csv(MASSLOSS_FILENAME)
         if 'snap_int' not in mass_loss_df.columns: # Ensure snap_int for min/max
             mass_loss_df['snap_int'] = mass_loss_df['snapshot'].astype(int)
     else:
         mass_loss_df = pd.DataFrame() 

if not mass_loss_df.empty:
     initial_mass_bound = mass_loss_df['Mstar_bound'].iloc[0]
     final_mass_bound = mass_loss_df['Mstar_bound'].iloc[-1]

     # M33_init_star_val should ideally be available from Cell 4 or Cell 6 logic
     # If not, it might be NaN here.
     if 'M33_init_star_val' not in locals() or np.isnan(M33_init_star_val):
         # Attempt one more time to get M33_init_star if it wasn't set
         print("Attempting to recalculate initial mass for summary (M33_init_star_val not found or NaN)...")
         # Use M33_init_star from Cell 4 if available
         if 'M33_init_star' in locals() and M33_init_star is not None and M33_init_star > 0:
             M33_init_star_val = M33_init_star
             print(f"Using M33_init_star from Cell 4: {M33_init_star_val:.2e} Msun")
         else: # Fallback to reading file if all else fails
             try:
                if 'snap_int' in mass_loss_df.columns and not mass_loss_df.empty:
                    zero_snap_sum = min(mass_loss_df['snap_int'])
                    if 'BASE_DATA_DIR' in locals() and 'Read' in locals():
                        init_file_sum = BASE_DATA_DIR / f"M33/M33_{zero_snap_sum:03d}.txt"
                        if init_file_sum.exists():
                             _, _, data0_sum = Read(str(init_file_sum))
                             if 'PTYPE_STARS' in locals():
                                  star_idx0_sum = np.isin(data0_sum['type'], PTYPE_STARS)
                                  M33_init_star_val = np.sum(data0_sum['m'][star_idx0_sum]) * 1e10
                                  if M33_init_star_val <= 0: M33_init_star_val = np.nan # bad calc
                             else: M33_init_star_val = np.nan
                        else: M33_init_star_val = np.nan
                    else: M33_init_star_val = np.nan
                else: M33_init_star_val = np.nan
             except Exception as e:
                print(f"Recalculation error for summary: {e}")
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
