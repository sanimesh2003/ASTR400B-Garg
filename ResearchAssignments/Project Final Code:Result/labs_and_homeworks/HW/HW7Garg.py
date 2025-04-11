 # Make edits where instructed - look for "****", which indicates where you need to add code. 
# ---------------------------- Cell 1 ---------------------------- #

# Homework 7 Template
# Rixin Li & G. Besla
#
# Make edits where instructed - look for "****", which indicates where you need to 
# add code. 

# import necessary modules
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as const

# import Latex module so we can display the results with symbols
from IPython.display import Latex

# **** import CenterOfMass to determine the COM pos/vel of M33
from CenterOfMass import CenterOfMass

# **** import the GalaxyMass to determine the mass of M31 for each component
from GalaxyMass import ComponentMass

# Cell 2

class M33AnalyticOrbit:
    """ Calculate the analytical orbit of M33 around M31 """
    
    def __init__(self, outfilename):
        """ **** ADD COMMENTS 
        
        Initialize the M33AnalyticOrbit class, setting up all relevant 
        parameters for M33's orbit about M31 using their positions and 
        velocities at snapshot 0.
        
        Inputs:
        -------
        outfilename : str
            The name of the output file where we'll save the integrated orbit.
        """
        
        ### get the gravitational constant (the value is 4.498768e-06 in kpc^3/Msun/Gyr^2)
        self.G = 4.498768e-6
        
        ### **** store the output file name
        self.filename = outfilename
        
        ### get the current pos/vel of M33 
        # **** create an instance of the  CenterOfMass class for M33 
        COM_M33 = CenterOfMass("M33_000.txt", 2)  # 2=disk
        posM33 = COM_M33.COM_P(delta=0.1) 
        velM33 = COM_M33.COM_V(posM33[0], posM33[1], posM33[2])
        
        # **** store the position VECTOR of the M33 COM (.value to get rid of units)
        posM33 = posM33.value
        velM33 = velM33.value
        
        ### get the current pos/vel of M31 
        COM_M31 = CenterOfMass("M31_000.txt", 2)
        posM31 = COM_M31.COM_P(delta=0.1)
        velM31 = COM_M31.COM_V(posM31[0], posM31[1], posM31[2])
        
        posM31 = posM31.value
        velM31 = velM31.value
        
        ### store the DIFFERENCE between the vectors posM33 - posM31
        # **** create two VECTORs self.r0 and self.v0 and have them be the
        # relative position and velocity VECTORS of M33
        self.r0 = posM33 - posM31
        self.v0 = velM33 - velM31
        
        
        ### get the mass of each component in M31 
        ### disk
        # **** self.rdisk = scale length (no units)
        self.rdisk = 5.0
        
        # **** self.Mdisk set with ComponentMass function. 
        #      * 1e12 to convert from 1e12 Msun to Msun 
        self.Mdisk  = 1e12*ComponentMass("M31_000.txt", 2)
        
        ### bulge
        # **** self.rbulge = set scale length (no units)
        self.rbulge = 1.0
        
        # **** self.Mbulge  set with ComponentMass function
        self.Mbulge = 1e12*ComponentMass("M31_000.txt", 3)
        
        ### Halo
        # **** self.rhalo = set scale length from HW5 (no units)
        # (the instructions say to use 60.0)
        self.rhalo = 60.0
        
        # **** self.Mhalo set with ComponentMass function
        self.Mhalo = 1e12*ComponentMass("M31_000.txt", 1)
    
    
    
    def HernquistAccel(self, M, ra, r):
        """ **** ADD COMMENTS 
        
        Computes the gravitational acceleration from a Hernquist profile:
        
            a = -G M / [r_mag (r_a + r_mag)^2] * r_vec
        
        Inputs:
        -------
        M   : float
            Total mass of the halo or bulge (Msun)
        ra  : float
            Scale radius (kpc)
        r   : np.ndarray of shape (3,)
            Position vector [x, y, z] in kpc
        
        Returns:
        --------
        accel : np.ndarray of shape (3,)
            Acceleration vector in kpc/Gyr^2
        """
        
        # store the magnitude of the position vector
        rmag = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
        
        # avoid divide-by-zero issues if rmag=0
        if rmag == 0:
            return np.array([0.0, 0.0, 0.0])
        
        # store the acceleration vector 
        # a = - G * M / [rmag * (ra + rmag)^2] * r
        Hern = - self.G * M / (rmag*(ra + rmag)**2) * r
        
        return Hern
    
    
    
    def MiyamotoNagaiAccel(self, M, rd, r):
        """ **** ADD COMMENTS 

        Computes the gravitational acceleration from a Miyamoto-Nagai disk profile:
        
        a = -G M / (R^2 + B^2)^(3/2) * r * [1, 1, B / sqrt(z^2 + z_d^2] 
        where 
            R = sqrt(x^2 + y^2)
            B = rd + sqrt(z^2 + z_d^2)
            z_d = rd/5.0
        """
        
        x, y, z = r
        # define R = sqrt(x^2 + y^2)
        R = np.sqrt(x**2 + y**2)
        
        # define z_d:
        z_d = rd/5.0
        # define B
        B = rd + np.sqrt(z**2 + z_d**2)
        
        # define (R^2 + B^2)^(3/2)
        denom = (R**2 + B**2)**1.5
        
        # again handle the case if we happen to be at r=0:
        if denom == 0:
            return np.array([0.0, 0.0, 0.0])
        
        # The standard formula is:
        # ax = -G M x / (R^2 + B^2)^(3/2)
        # ay = -G M y / (R^2 + B^2)^(3/2)
        # az = -G M B z / [ (R^2 + B^2)^(3/2) sqrt(z^2 + z_d^2) ]
        
        # We can do this in a vector form as in eqn (4) in the instructions:
        # a = -GM / (R^2 + B^2)^(3/2) * r * [1, 1, B/sqrt(z^2 + z_d^2)]
        
        # define the factor -G M / (R^2 + B^2)^(3/2)
        factor = - self.G * M / denom
        
        # the z correction factor:
        zfactor = B / np.sqrt(z**2 + z_d**2)
        
        # build a small array to multiply with r:
        multi = np.array([1.0, 1.0, zfactor])
        
        # the final acceleration vector:
        a_disk = factor * r * multi
        
        return a_disk
    
    
    def M31Accel(self, r):
        """ **** ADD COMMENTS 
        
        Computes the total acceleration from M31's halo, bulge, and disk 
        at the position vector r.
        
        Inputs:
        -------
        r : np.ndarray of shape (3,)
            Relative position vector in kpc

        Returns:
        --------
        a_total : np.ndarray of shape (3,)
            Sum of all acceleration components from M31: halo+bulge+disk
        """
        
        # Call the HernquistAccel for the halo
        a_halo  = self.HernquistAccel(self.Mhalo,  self.rhalo,  r)
        
        # Call the HernquistAccel for the bulge
        a_bulge = self.HernquistAccel(self.Mbulge, self.rbulge, r)
        
        # Call the MiyamotoNagaiAccel for the disk
        a_disk  = self.MiyamotoNagaiAccel(self.Mdisk, self.rdisk, r)
        
        # Sum them up (vector sum)
        a_total = a_halo + a_bulge + a_disk
        
        return a_total
    
    
    def LeapFrog(self, r, v, dt):
        """
        Advance the position and velocity of M33 by one timestep dt 
        using the Leap Frog integrator.
        
        Parameters
        ----------
        r : np.ndarray
            Current position vector [x, y, z] of M33 relative to M31
        v : np.ndarray
            Current velocity vector [vx, vy, vz] of M33 relative to M31
        dt : float
            Timestep in Gyr; can be positive or negative
        
        Returns
        -------
        rnew : np.ndarray
            Updated position vector after one timestep
        vnew : np.ndarray
            Updated velocity vector after one timestep
        
        The integration scheme:
        
        1) Predict the half-step position:
            r_half = r + v*(dt/2)
            
        2) Compute acceleration a_half at r_half:
            a_half = M31Accel(r_half)
            
        3) Update the velocity at the full step:
            v_new = v + a_half*dt
            
        4) Update the position at the full step 
           (using r_half and v_new):
            r_new = r_half + v_new*(dt/2)
        """
        
        # 1) Predict the half-step position
        rhalf = r + v*(dt/2.0)
        
        # 2) Compute the acceleration at the half-step
        a_half = self.M31Accel(rhalf)
        
        # 3) Update velocity to the next full step
        vnew = v + a_half*dt
        
        # 4) Update position to the next full step
        rnew = rhalf + vnew*(dt/2.0)
        
        return rnew, vnew
    
    def OrbitIntegration(self, t0, dt, tmax):
        """
        Integrate M33's orbit forward in time from t0 to tmax using 
        the LeapFrog method defined in LeapFrog().
        
        Parameters
        ----------
        t0   : float
            Initial time in Gyr (e.g., 0.0)
        dt   : float
            Timestep in Gyr (e.g., 0.01)
        tmax : float
            Final time in Gyr (e.g., 10.0)
        
        Returns
        -------
        None. The orbit array is written to self.filename as a txt file
        with columns: [t, x, y, z, vx, vy, vz].
        """
        
        # initialize the time
        t = t0
        
        # estimate how many steps we need
        # e.g. if dt=0.01 Gyr and tmax=10 Gyr, we might have 1000 steps
        # but tmax can be anything, so let's use int(...) plus some buffer
        nsteps = int( (tmax - t0)/dt ) + 2
        
        # initialize an array to store time, x, y, z, vx, vy, vz
        # shape: (nsteps, 7)
        orbit = np.zeros( (nsteps, 7) )
        
        # current position and velocity from the initialization
        rcurrent = self.r0
        vcurrent = self.v0
        
        # store the initial conditions in the first row
        # orbit[0] = t0, rx, ry, rz, vx, vy, vz
        orbit[0,0] = t
        orbit[0,1:4] = rcurrent
        orbit[0,4:7] = vcurrent
        
        # index counter for array
        i = 1
        
        # start the integration loop
        while (t < tmax) and (i < nsteps):
            
            # advance time by dt
            t += dt
            
            # use LeapFrog to get new position and velocity
            rnew, vnew = self.LeapFrog(rcurrent, vcurrent, dt)
            
            # store the new time and phase-space coordinates
            orbit[i,0]   = t
            orbit[i,1:4] = rnew
            orbit[i,4:7] = vnew
            
            # update rcurrent, vcurrent for next iteration
            rcurrent = rnew
            vcurrent = vnew
            
            # increment index
            i += 1
        
        # now that we've finished, let's trim the orbit array 
        # in case we didn't use all rows:
        orbit = orbit[:i]
        
        # write the array to file
        # e.g. columns: t, x, y, z, vx, vy, vz
        np.savetxt(self.filename, orbit, fmt="%11.3f", comments='#',
                   header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}"
                   .format('t','x','y','z','vx','vy','vz'))
        
# Return the final orbit array
M33orbit = M33AnalyticOrbit("M33_analytic.txt")
M33orbit.OrbitIntegration(t0=0.0, dt=0.01, tmax=10.0)
print('File created')

# ----------------- Code for Part 5 question 1 ----------------- #

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1) Read ANALYTIC orbit
# ------------------------------
data_analytic = np.genfromtxt("M33_analytic.txt", comments='#')
# columns: t, x, y, z, vx, vy, vz
tA  = data_analytic[:,0]
xA  = data_analytic[:,1]
yA  = data_analytic[:,2]
zA  = data_analytic[:,3]
vxA = data_analytic[:,4]
vyA = data_analytic[:,5]
vzA = data_analytic[:,6]

rA = np.sqrt(xA**2 + yA**2 + zA**2)    # total distance
vA = np.sqrt(vxA**2 + vyA**2 + vzA**2) # total speed

# ------------------------------
# 2) Read SIMULATED orbits from HW6
#    We want M33 relative to M31
# ------------------------------
dataM31 = np.genfromtxt("Orbit_M31.txt", comments='#')
dataM33 = np.genfromtxt("Orbit_M33.txt", comments='#')
# columns: t, x, y, z, vx, vy, vz

t_M31  = dataM31[:,0]
x_M31  = dataM31[:,1]
y_M31  = dataM31[:,2]
z_M31  = dataM31[:,3]
vx_M31 = dataM31[:,4]
vy_M31 = dataM31[:,5]
vz_M31 = dataM31[:,6]

t_M33  = dataM33[:,0]
x_M33  = dataM33[:,1]
y_M33  = dataM33[:,2]
z_M33  = dataM33[:,3]
vx_M33 = dataM33[:,4]
vy_M33 = dataM33[:,5]
vz_M33 = dataM33[:,6]

x_rel = x_M33 - x_M31
y_rel = y_M33 - y_M31
z_rel = z_M33 - z_M31

vx_rel = vx_M33 - vx_M31
vy_rel = vy_M33 - vy_M31
vz_rel = vz_M33 - vz_M31

rSim = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)    
vSim = np.sqrt(vx_rel**2 + vy_rel**2 + vz_rel**2)

tSim = t_M33  # same as t_M31

# ------------------------------
# 3) Plot the results
# ------------------------------
fig, axes = plt.subplots(1,2, figsize=(16,6))

# Left: distance vs time
axes[0].plot(tA,  rA,  label='M33-M31 predicted',  color='blue')
axes[0].plot(tSim, rSim, label='M33-M31 simulation', color='red')
axes[0].set_xlabel('Time (Gyr)', fontsize=14)
axes[0].set_ylabel('Distance (kpc)', fontsize=14)
axes[0].set_title('M33 Distance from M31', fontsize=15)
axes[0].legend()

# Right: speed vs time
axes[1].plot(tA,  vA,  label='M33-M31 predicted', color='blue')
axes[1].plot(tSim, vSim, label='M33-M31 simulation', color='red')
axes[1].set_xlabel('Time (Gyr)', fontsize=14)
axes[1].set_ylabel('Speed (km/s)', fontsize=14)
axes[1].set_title('M33 Velocity relative to M31', fontsize=15)
axes[1].legend()

plt.tight_layout()
plt.show()

# analysis for Part 5 question 2 - this was just for a visual representation nothing else #

import numpy as np
import matplotlib.pyplot as plt

# 1) Read in or define the arrays from your existing code

rA_interp = np.interp(tSim, tA, rA)
vA_interp = np.interp(tSim, tA, vA)

# 2) Compute the difference or ratio
r_diff  = rSim - rA_interp
v_diff  = vSim - vA_interp

# or relative difference
# e.g. ratio = (Sim - Ana) / Ana
r_ratio = r_diff / rA_interp
v_ratio = v_diff / vA_interp

# 3) Plot the differences vs time
fig, ax = plt.subplots(1, 2, figsize=(14,5))

# Left: Distance difference
ax[0].plot(tSim, r_diff, 'b', label='Distance difference (kpc)')
ax[0].axhline(0, color='k', ls='--', lw=1)
ax[0].set_xlabel('Time (Gyr)', fontsize=14)
ax[0].set_ylabel('r_Sim - r_Ana (kpc)', fontsize=14)
ax[0].legend()

# Right: Velocity difference
ax[1].plot(tSim, v_diff, 'r', label='Velocity difference (km/s)')
ax[1].axhline(0, color='k', ls='--', lw=1)
ax[1].set_xlabel('Time (Gyr)', fontsize=14)
ax[1].set_ylabel('v_Sim - v_Ana (km/s)', fontsize=14)
ax[1].legend()

plt.tight_layout()
plt.show()

# 4) Print some summary statistics
print("SUMMARY STATISTICS (Sim - Analytic):")
print(f"Distance difference: mean={np.mean(r_diff):.3f} kpc,  std={np.std(r_diff):.3f} kpc")
print(f"Velocity difference: mean={np.mean(v_diff):.3f} km/s, std={np.std(v_diff):.3f} km/s")

# You could also check max difference:
print(f"Max distance difference = {np.max(np.abs(r_diff)):.3f} kpc at t={tSim[np.argmax(np.abs(r_diff))]:.2f} Gyr")
print(f"Max velocity difference = {np.max(np.abs(v_diff)):.3f} km/s at t={tSim[np.argmax(np.abs(v_diff))]:.2f} Gyr")


