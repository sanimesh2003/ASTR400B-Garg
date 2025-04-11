# Cell 1

# Homework 6 Template
# G. Besla & R. Li

#cell 2

# import modules
import numpy as np
import astropy.units as u
from astropy.constants import G

# import plotting modules
import matplotlib.pyplot as plt
import matplotlib
#%matplotlib inline

# my modules
from ReadFile import Read
# Step 1: modify CenterOfMass so that COM_P now takes a parameter specifying 
# by how much to decrease RMAX instead of a factor of 2
from CenterOfMass2 import CenterOfMass

#cell 3

def OrbitCOM(galaxy, start, end, n):
    """
    function that loops over all the desired snapshots to compute the COM pos and vel 
    as a function of time.
    
    inputs:
    -------
    galaxy : str
        'MW', 'M31', 'M33'
    start  : int
        snapshot to start from
    end    : int
        snapshot to end on
    n      : int
        step size for reading snapshots
        
    outputs:
    --------
    A file "Orbit_<galaxy>.txt" containing columns:
    t, x, y, z, vx, vy, vz
    
    We only do this once and store the data so we don't 
    keep repeating this process.
    """

    # compose the filename for output
    fileout = f"Orbit_{galaxy}.txt"
    
    # set tolerance and volDec
    # For MW, M31 we can use volDec=2; for M33, use volDec=4 (since M33 is more stripped)
    delta = 0.1
    if galaxy == "M33":
        volDec = 4.0
    else:
        volDec = 2.0

    # generate the snapshot id sequence 
    snap_ids = np.arange(start, end+1, n)  # e.g. 0, 5, 10, ... 800
    # simple check:
    if len(snap_ids) == 0:
        print("No snapshots found. Check your inputs.")
        return

    # initialize the array for orbital info: t, x, y, z, vx, vy, vz
    orbit = np.zeros((len(snap_ids), 7))

    # a for loop to loop over files
    for i, snap_id in enumerate(snap_ids):
        
        # compose the data filename 
        # if using local directories named MW, M31, M33 for the files, do:
        ilbl = f"{snap_id:03d}"   # ensures a three-digit label
        filename = f"{galaxy}/{galaxy}_{ilbl}.txt"

        # Initialize an instance of CenterOfMass class, using DISK particles
        COM = CenterOfMass(filename, 2)

        # Store the COM pos and vel. COM_P now has volDec
        com_pos = COM.COM_P(delta=delta, volDec=volDec)
        com_vel = COM.COM_V(com_pos[0], com_pos[1], com_pos[2])
        
        # store the time, pos, vel in ith element of the orbit array,  without units (.value)
        # time is in Myr in the snapshot, so dividing by 1000 to get Gyr
        orbit[i, 0] = COM.time.to(u.Gyr).value
        orbit[i, 1] = com_pos[0].value # x
        orbit[i, 2] = com_pos[1].value # y
        orbit[i, 3] = com_pos[2].value # z
        orbit[i, 4] = com_vel[0].value # vx
        orbit[i, 5] = com_vel[1].value # vy
        orbit[i, 6] = com_vel[2].value # vz
        
        # print snap_id to see the progress
        print(f"Done with snapshot {snap_id}")

    # write the data to a file
    # we do this because we don't want to have to repeat this process 
    # this code should only have to be called once per galaxy.
    np.savetxt(fileout, orbit, fmt = "%11.3f"*7, comments='#',
               header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}"\
                      .format('t', 'x', 'y', 'z', 'vx', 'vy', 'vz'))
    print(f"Orbit saved to {fileout}")
    
#Cell 4

# Recover the orbits and generate the COM files for each galaxy
# read in 800 snapshots in intervals of n=5
# Note: This might take a little while - test your code with a smaller number of snapshots first! 

OrbitCOM("MW", 0, 800, 5)
OrbitCOM("M31", 0, 800, 5)
OrbitCOM("M33", 0, 800, 5)

#cell 5

# Read in the data files for the orbits of each galaxy that you just created
# headers:  t, x, y, z, vx, vy, vz
# using np.genfromtxt

import numpy as np
import matplotlib.pyplot as plt

dataMW  = np.genfromtxt("Orbit_MW.txt", comments='#')
dataM31 = np.genfromtxt("Orbit_M31.txt", comments='#')
dataM33 = np.genfromtxt("Orbit_M33.txt", comments='#')

# columns: t, x, y, z, vx, vy, vz
t_MW  = dataMW[:, 0]
x_MW  = dataMW[:, 1]
y_MW  = dataMW[:, 2]
z_MW  = dataMW[:, 3]
vx_MW = dataMW[:, 4]
vy_MW = dataMW[:, 5]
vz_MW = dataMW[:, 6]

t_M31  = dataM31[:, 0]
x_M31  = dataM31[:, 1]
y_M31  = dataM31[:, 2]
z_M31  = dataM31[:, 3]
vx_M31 = dataM31[:, 4]
vy_M31 = dataM31[:, 5]
vz_M31 = dataM31[:, 6]

t_M33  = dataM33[:, 0]
x_M33  = dataM33[:, 1]
y_M33  = dataM33[:, 2]
z_M33  = dataM33[:, 3]
vx_M33 = dataM33[:, 4]
vy_M33 = dataM33[:, 5]
vz_M33 = dataM33[:, 6]

#cell 6 

# function to compute the magnitude of the difference between two vectors 
# You can use this function to return both the relative position and relative velocity for two 
# galaxies over the entire orbit  

# function to compute the magnitude of the difference between two vectors 
def VectorDiffMag(x1, y1, z1, x2, y2, z2):
    """
    Returns the magnitude of the difference between
    vector1(x1, y1, z1) and vector2(x2, y2, z2).
    """
    return np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2 )

def VectorDiffMagVel(vx1, vy1, vz1, vx2, vy2, vz2):
    """
    Returns the magnitude of velocity difference
    between vector1(vx1, vy1, vz1) and vector2(vx2, vy2, vz2).
    """
    return np.sqrt( (vx2 - vx1)**2 + (vy2 - vy1)**2 + (vz2 - vz1)**2 )

# We assume t_MW, t_M31, t_M33 have the same shape 
# (they should, if they were written out the same way). 
# But let's do a quick check or at least assume they match.

#Cell 7

# Determine the magnitude of the relative position and velocities 

# of MW and M31
r_MW_M31  = VectorDiffMag(x_MW,  y_MW,  z_MW,  x_M31,  y_M31,  z_M31)
v_MW_M31  = VectorDiffMagVel(vx_MW, vy_MW, vz_MW, vx_M31, vy_M31, vz_M31)

# of M33 and M31
r_M33_M31 = VectorDiffMag(x_M33, y_M33, z_M33, x_M31, y_M31, z_M31)
v_M33_M31 = VectorDiffMagVel(vx_M33, vy_M33, vz_M33, vx_M31, vy_M31, vz_M31)

#cell 8

# Plot the Orbit of the galaxies 
#################################
# Plot the orbital velocities of the galaxies 
#################################

fig, ax = plt.subplots(1,2, figsize=(25,10))

# Left subplot: separation vs time
ax[0].plot(t_MW, r_MW_M31, color='blue',  label='MW-M31')
ax[0].plot(t_M31, r_M33_M31, color='red', label='M33-M31')
ax[0].set_xlabel('Time (Gyr)')
ax[0].set_ylabel('Separation (kpc)')
ax[0].set_title('Separation vs Time')
ax[0].legend()

# Right subplot: relative speed vs time
ax[1].plot(t_MW, v_MW_M31, color='blue',  label='MW-M31')
ax[1].plot(t_M31, v_M33_M31, color='red', label='M33-M31')
ax[1].set_xlabel('Time (Gyr)')
ax[1].set_ylabel('Relative Speed (km/s)')
ax[1].set_title('Relative Velocity vs Time')
ax[1].legend()

plt.tight_layout()
plt.show()

# Question 4A : Identify local minima in r_MW_M31 and count only MAJOR close encounters

def find_local_minima(x):
    """
    Returns a list of indices i where x[i] is a local minimum:
    x[i] < x[i-1] and x[i] < x[i+1].
    """
    minima_indices = []
    for i in range(1, len(x)-1):
        if (x[i] < x[i-1]) and (x[i] < x[i+1]):
            minima_indices.append(i)
    return minima_indices

def find_local_maxima(x):
    """
    Returns a list of indices i where x[i] is a local maximum:
    x[i] > x[i-1] and x[i] > x[i+1].
    """
    maxima_indices = []
    for i in range(1, len(x)-1):
        if (x[i] > x[i-1]) and (x[i] > x[i+1]):
            maxima_indices.append(i)
    return maxima_indices

# Find all local minima in separation
r_min_indices = find_local_minima(r_MW_M31)
r_max_indices = find_local_maxima(r_MW_M31)  # To find previous peak before dip

# Define lower and upper bounds for close encounters
lower_thresh = 1     # Encounters must be closer than this
upper_thresh = 50    # Ignore separations above this
major_dip = 1       # Must be at least 1 kpc lower than previous peak

filtered_encounters = []

# Loop through minima and check criteria
for idx in r_min_indices:
    if lower_thresh <= r_MW_M31[idx] <= upper_thresh:  # Within range
        # Find the most recent maximum before this minimum
        prev_max_idx = max([m for m in r_max_indices if m < idx], default=None)

        if prev_max_idx is not None:
            drop = r_MW_M31[prev_max_idx] - r_MW_M31[idx]  # How much separation dropped
            if drop >= major_dip:  # Only count if drop is significant
                filtered_encounters.append(idx)

# Print the results
print(f"\nMajor Close Encounters (1 < r < 50 kpc & Δr >= {major_dip} kpc):")
for i, idx in enumerate(filtered_encounters):
    print(f"{i+1}) t={t_MW[idx]:.2f} Gyr, r={r_MW_M31[idx]:.2f} kpc")

print(f"\nTotal number of major encounters = {len(filtered_encounters)}")

# '''
# Question 4A
#
# visually its 3 close encounters before it then goes off to zero eventually with changes that this code shows 
# and the output reveals more encounters.
#
# the issue that you see with my code is while it is able to pick up absolute minute details in the shifts of separation 
# because of the ability to adjust threshold, you are not yet able to tell the code that you want it to select only 
# major points of cross over which is why it reveals 9 points.
# 
# I am working on updating this so that you can set a threshhold upper and lower limit to try and see what options 
# are available.
#
# Update - tried a new technique for the thresholding that has worked and now the code also shows 3 close encounters.
# 
# '''
# Question 4B : Relationship between separation & relative velocity

# ''' 
# Question 4B
# 
# they are inversely proportianate to each other as you can see that in the relation of MW and M31, as the separation comes to a sudden decrease you see a spike in the velocity graph skyrocket.
# then as the separation gets further you see that the velocity slows down again, and then repeats that process till they eventually come to a natural zero point where they are within close enough
# distance to each other that you then see that they separation and the velocity are approaching 0, which in this case is still incerdibly fast.
# 
# this is different for the relation of M33 and M31, where as you can see the shifting of the 2 galaxies coming together also causes them to go out far enough as well. however what you are able to
# see again is the inverse proportion to the lines themselves which goes to further prove the point discussed above.
# 
# '''
# Question 4C : Checking final separation + zoomed/log plot for MW-M31

import numpy as np
import matplotlib.pyplot as plt

# 1) Check final separation
final_sep = r_MW_M31[-1]
final_time = t_MW[-1]
print(f"Final snapshot time ~ {final_time:.2f} Gyr, separation = {final_sep:.2f} kpc")

# If you want to see if/when r < 10 kpc at any time:
merge_idx = np.where(r_MW_M31 < 10)[0]
if len(merge_idx) > 0:
    print("Times when MW-M31 separation < 10 kpc:")
    for i in merge_idx:
        print(f"  t={t_MW[i]:.3f} Gyr, r={r_MW_M31[i]:.3f} kpc")
else:
    print("No times found with MW-M31 separation < 10 kpc in our snapshots.")

# 2) Zoomed plot: we can focus on t>4 Gyr or t>5 Gyr for a better look
zoom_mask = np.where(t_MW > 4.0)[0]

fig, ax = plt.subplots(1,2, figsize=(14,5))

# Left: normal scale
ax[0].plot(t_MW[zoom_mask], r_MW_M31[zoom_mask], label='MW-M31')
ax[0].plot(t_M31[zoom_mask], r_M33_M31[zoom_mask], label='M33-M31')
ax[0].set_xlabel('Time (Gyr)')
ax[0].set_ylabel('Separation (kpc)')
ax[0].set_title('Zoomed: t > 4 Gyr')
ax[0].legend()

# Right: log scale on y
ax[1].semilogy(t_MW[zoom_mask], r_MW_M31[zoom_mask], label='MW-M31')
ax[1].semilogy(t_M31[zoom_mask], r_M33_M31[zoom_mask], label='M33-M31')
ax[1].set_xlabel('Time (Gyr)')
ax[1].set_ylabel('Separation (kpc) [log scale]')
ax[1].set_title('Log scale: t > 4 Gyr')
ax[1].legend()

plt.tight_layout()
plt.show()

# 3) Checking M33’s orbit after that merge time:
# e.g. if the last close approach is near t~5 Gyr, let's see r_M33_M31 after that
merge_time = 5.0  # example
m33_indices_after_merge = np.where(t_M33 >= merge_time)[0]
print(f"Average M33-M31 separation after t={merge_time} Gyr: "
      f"{np.mean(r_M33_M31[m33_indices_after_merge]):.2f} kpc")


# '''
# Question 4C
# 
# Final snapshot time ~ 11.43 Gyr, separation = ~1.43 kpc
# Times when MW-M31 separation < 10 kpc: (long list)
# Average M33-M31 separation after t=5.0 Gyr: ~83.21 kpc
# 
# The figure also shows that around 5-6 Gyr we see them bouncing 
# below 10 kpc multiple times, effectively merging by ~6-8 Gyr 
# and staying near 1-2 kpc from ~10 Gyr onward. M33 remains 
# tens of kpc away, with repeated orbits.
# 
# '''
# Question 4D : M33 orbit decay rate after 6 Gyr

# We'll find apocenters for M33-M31 after t=6 Gyr. Apocenters = local maxima
def find_local_maxima(x):
    """Return indices i where x[i] is a local maximum."""
    maxima_indices = []
    for i in range(1, len(x)-1):
        if (x[i] > x[i-1]) and (x[i] > x[i+1]):
            maxima_indices.append(i)
    return maxima_indices

t_cut = 6.0
idx_6Gyr = np.where(t_M33 >= t_cut)[0]
apocenter_indices = []

# We'll only search local maxima in the portion t>=6 Gyr
# so we limit ourselves to the range [idx_6Gyr[0], len(r_M33_M31)-2]
start_i = idx_6Gyr[0]
end_i   = len(r_M33_M31) - 1
for i in range(start_i+1, end_i):
    if (r_M33_M31[i] > r_M33_M31[i-1]) and (r_M33_M31[i] > r_M33_M31[i+1]):
        apocenter_indices.append(i)

print("M33 Apocenters after 6 Gyr:")
for i in apocenter_indices:
    print(f"  t={t_M33[i]:.2f} Gyr, r={r_M33_M31[i]:.2f} kpc")

# If we have at least 2 apocenters, measure the difference
if len(apocenter_indices) >= 2:
    i1 = apocenter_indices[0]
    i2 = apocenter_indices[1]
    rA1 = r_M33_M31[i1]
    rA2 = r_M33_M31[i2]
    tA1 = t_M33[i1]
    tA2 = t_M33[i2]

    dr = rA1 - rA2
    dt = tA2 - tA1
    decay_rate = dr/dt  # kpc/Gyr
    print(f"\nApprox. decay rate between first two apocenters: {decay_rate:.2f} kpc/Gyr")

    # Estimate how long M33 would take to merge if it starts ~75 kpc away
    # time = distance / rate
    # But if rate is negative => r is decreasing
    future_dist = 75.0
    time_to_merge = future_dist / decay_rate if decay_rate>0 else None
    if time_to_merge and time_to_merge>0:
        print(f"Estimated time for M33 to merge from 75 kpc: {time_to_merge:.2f} Gyr")
    else:
        print("Decay rate is negative or zero -- can't estimate a straightforward time to merge.")
else:
    print("Not enough apocenters after 6 Gyr to estimate a decay rate.")

# '''
# Question 4D
# 
# M33 Apocenters after 6 Gyr:
#   t=7.50 Gyr, r=108.76 kpc
#   t=8.93 Gyr, r=89.08 kpc
#   t=10.07 Gyr, r=77.42 kpc
#   t=11.07 Gyr, r=70.71 kpc
# 
# Approx. decay rate between first two apocenters: 13.77 kpc/Gyr
# Estimated time for M33 to merge from 75 kpc: 5.45 Gyr
# 
# '''

