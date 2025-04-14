import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import os
import pandas as pd
import glob
from scipy.optimize import curve_fit

class MassProfile:
    """
    Class that computes the mass profile for a snapshot of a galaxy,
    given the galaxy's COM position, so we can measure distances from COM.
    """
    def __init__(self, data, xCOM, yCOM, zCOM):
        """
        PARAMETERS
        ----------
        data : np.ndarray
            Structured array with columns 'type', 'm', 'x', 'y', 'z', ...
        xCOM, yCOM, zCOM : float
            The center-of-mass (in kpc) for this galaxy
        """
        self.data = data  # entire structured array
        self.xCOM = xCOM
        self.yCOM = yCOM
        self.zCOM = zCOM

        # G in convenient units: kpc (km/s)^2 / Msun
        self.G = 4.498768e-6

    def _distance_from_com(self, x, y, z):
        """
        Internal helper function to compute 3D distance from COM
        for each particle. Returns array of distances in kpc.
        """
        dx = x - self.xCOM
        dy = y - self.yCOM
        dz = z - self.zCOM
        rr = np.sqrt(dx*dx + dy*dy + dz*dz)
        return rr

    def MassEnclosed(self, ptype, radius):
        """
        Compute the enclosed mass for a specific ptype (1=Halo,2=Disk,3=Bulge)
        within a given radius (or array of radii).

        PARAMETERS
        ----------
        ptype : int
            Particle type (1,2,3)
        radius : float or array-like
            Single radius in kpc, or an array of radii in kpc

        RETURNS
        -------
        Menc : float or np.ndarray
            The enclosed mass in Msun, same shape as 'radius'
        """
        # Filter the data for just that ptype
        idx = np.where(self.data['type'] == ptype)[0]
        mass_arr = self.data['m'][idx]  # in 1e10 Msun
        x_arr = self.data['x'][idx]
        y_arr = self.data['y'][idx]
        z_arr = self.data['z'][idx]

        rr = self._distance_from_com(x_arr, y_arr, z_arr)

        r_array = np.atleast_1d(radius)

        Menc_list = []
        for rmax in r_array:
            inside_idx = np.where(rr < rmax)[0]
            mass_enclosed = np.sum(mass_arr[inside_idx]) * 1e10  # Msun
            Menc_list.append(mass_enclosed)

        if len(Menc_list) == 1:
            return Menc_list[0]
        else:
            return np.array(Menc_list)

    def MassEnclosedTotal(self, radius):
        """
        Sum of halo, disk, and bulge within 'radius'.
        (We assume ptype=1,2,3 are the only ones.)

        RETURNS
        -------
        Mtot : float or array, Msun
        """
        M1 = self.MassEnclosed(1, radius)  # halo
        M2 = self.MassEnclosed(2, radius)  # disk
        M3 = self.MassEnclosed(3, radius)  # bulge if it exists
        return M1 + M2 + M3
