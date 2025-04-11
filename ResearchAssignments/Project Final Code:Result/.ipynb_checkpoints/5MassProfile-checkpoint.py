import numpy as np
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt

# Import your Read and CenterOfMass functions
from ReadFile import Read
from CenterOfMass import CenterOfMass

class MassProfile:
    """
    A class to calculate mass profiles and rotation curves for a given galaxy at a given snapshot.
    """

    def __init__(self, galaxy, snap):
        """
        Initialize the class by reading in data from a snapshot file, and 
        computing the center of mass from disk particles.

        Parameters
        ----------
        galaxy : str
            The name of the galaxy ('MW', 'M31', or 'M33').
        snap : int
            The snapshot number, e.g., 0 for present day.
        """

        # Store galaxy name as a property
        self.gname = galaxy
        
        # Construct the filename. Example: 'MW_000.txt' if galaxy='MW' and snap=0
        ilbl = '000' + str(snap)
        ilbl = ilbl[-3:]
        self.filename = f"{galaxy}_{ilbl}.txt"
        
        # Read the data from this file
        self.time, self.total, self.data = Read(self.filename)
        
        # Store positions and masses in astropy units
        self.x = self.data['x'] * u.kpc
        self.y = self.data['y'] * u.kpc
        self.z = self.data['z'] * u.kpc
        self.m = self.data['m'] * 1e10 * u.Msun  # Msun
        
        # Compute the galaxy's COM position using DISK particles
        com_object = CenterOfMass(self.filename, 2)  # 2 = disk
        self.com_pos = com_object.COM_P(0.1)  # returns [x_COM, y_COM, z_COM] in kpc
        
        # Store G in convenient units: kpc (km/s)^2 / Msun
        self.G = G.to(u.kpc * u.km**2 / (u.s**2 * u.Msun))


    def MassEnclosed(self, ptype, radii):
        """
        Computes the mass enclosed within each radius in 'radii' (array) for particles of type 'ptype'.
        
        Parameters
        ----------
        ptype : int
            Particle type: 1 (Halo), 2 (Disk), 3 (Bulge).
        radii : np.ndarray or list of floats
            Radii in kpc at which to compute enclosed mass.
        
        Returns
        -------
        enclosed_mass : astropy.units.Quantity
            Array of enclosed masses at each radius in Msun.
        """
        # Select only particles of the requested type
        index = np.where(self.data['type'] == ptype)
        m_selected = self.m[index]  # Msun
        x_selected = self.x[index]  # kpc
        y_selected = self.y[index]  # kpc
        z_selected = self.z[index]  # kpc
        
        # Center relative to COM
        dx = x_selected - self.com_pos[0]
        dy = y_selected - self.com_pos[1]
        dz = z_selected - self.com_pos[2]
        r_part = np.sqrt(dx**2 + dy**2 + dz**2)  # distance of each particle from COM
        
        # Ensure 'radii' is an astropy quantity in kpc
        radii_kpc = radii * u.kpc
        
        # Initialize array for enclosed mass
        enclosed_mass = np.zeros(len(radii_kpc)) * u.Msun
        
        # Loop over the input radii
        for i in range(len(radii_kpc)):
            index_within = np.where(r_part < radii_kpc[i])
            enclosed_mass[i] = np.sum(m_selected[index_within])
        
        return enclosed_mass


    def MassEnclosedTotal(self, radii):
        """
        Computes the total enclosed mass (halo+disk+bulge) for each radius in 'radii'.

        Parameters
        ----------
        radii : array-like
            Radii in kpc.

        Returns
        -------
        total_mass : astropy.units.Quantity
            Array of total enclosed mass at each radius in Msun.
        """
        # Halo + Disk
        m_halo = self.MassEnclosed(1, radii)
        m_disk = self.MassEnclosed(2, radii)
        
        # Bulge (unless M33, which has no bulge)
        if self.gname == 'M33':
            m_bulge = 0 * m_halo
        else:
            m_bulge = self.MassEnclosed(3, radii)
        
        total_mass = m_halo + m_disk + m_bulge
        return total_mass


    def HernquistMass(self, r, a, Mhalo):
        """
        Computes the Hernquist 1990 mass profile: M(r) = Mhalo * (r^2 / (r+a)^2).
        
        Parameters
        ----------
        r : float or np.ndarray
            Radius (kpc)
        a : float
            Scale radius (kpc)
        Mhalo : float
            Total halo mass in Msun
        
        Returns
        -------
        M_hern : astropy.units.Quantity
            Enclosed mass at radius r in Msun.
        """
        r_kpc = r * u.kpc
        a_kpc = a * u.kpc
        Mhalo_msun = Mhalo * u.Msun
        
        frac = (r_kpc**2) / (r_kpc + a_kpc)**2
        M_hern = Mhalo_msun * frac
        return M_hern


    def CircularVelocity(self, ptype, radii):
        """
        Computes the circular velocity for the given component at each radius in 'radii'.
        
        Vc(r) = sqrt(G * M_enclosed(r) / r)
        
        Parameters
        ----------
        ptype : int
            Particle type (1=Halo, 2=Disk, 3=Bulge).
        radii : np.ndarray
            Radii in kpc.
        
        Returns
        -------
        V_circ : astropy.units.Quantity
            Circular velocity in km/s (rounded to 2 decimals).
        """
        # Enclosed mass
        Menc = self.MassEnclosed(ptype, radii)  # Msun
        r_kpc = radii * u.kpc
        
        V_circ = np.sqrt(self.G * Menc / r_kpc)
        return np.round(V_circ, 2)


    def CircularVelocityTotal(self, radii):
        """
        Computes the total circular velocity (halo + disk + bulge) at each radius in 'radii'.
        
        V_circ_total(r) = sqrt(G * [M_halo + M_disk + M_bulge] / r)
        
        Parameters
        ----------
        radii : array-like
            Radii in kpc.
        
        Returns
        -------
        V_circ_total : astropy.units.Quantity
            Total circular velocity in km/s (rounded to 2 decimals).
        """
        Mtot = self.MassEnclosedTotal(radii)  # Msun
        r_kpc = radii * u.kpc
        
        V_circ = np.sqrt(self.G * Mtot / r_kpc)
        return np.round(V_circ, 2)


    def HernquistVCirc(self, r, a, Mhalo):
        """
        Computes the circular velocity for a Hernquist halo:
        M(r) = Mhalo * (r^2 / (r+a)^2).
        V_circ(r) = sqrt(G * M(r) / r).
        
        Parameters
        ----------
        r : float or array-like
            Radius in kpc.
        a : float
            Scale radius in kpc.
        Mhalo : float
            Halo mass in Msun.
        
        Returns
        -------
        V_hern : astropy.units.Quantity
            Hernquist circular velocity in km/s (rounded to 2 decimals).
        """
        r_kpc = r * u.kpc
        a_kpc = a * u.kpc
        Mhalo_msun = Mhalo * u.Msun
        
        # M(r) for Hernquist
        M_hern = Mhalo_msun * (r_kpc**2 / (r_kpc + a_kpc)**2)
        
        V_hern = np.sqrt(self.G * M_hern / r_kpc)
        return np.round(V_hern, 2)
