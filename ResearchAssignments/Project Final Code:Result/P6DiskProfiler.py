import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from P1Read import Read

class DiskProfiler:
    def __init__(self, com_file="All_COM.csv", data_dir="M33/", output_file="M33_DiskProfileFits.csv"):
        self.com_file = com_file
        self.data_dir = data_dir
        self.output_file = output_file

    @staticmethod
    def rotate_to_face_on(positions, velocities):
        L = np.sum(np.cross(positions, velocities), axis=0)
        Lmag = np.linalg.norm(L)
        if Lmag == 0:
            return positions, velocities
        Lhat = L / Lmag

        zhat = np.array([0, 0, 1.0])
        v = np.cross(Lhat, zhat)
        s = np.linalg.norm(v)
        c = np.dot(Lhat, zhat)
        if s == 0:
            return positions, velocities

        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)
        return positions @ R.T, velocities @ R.T

    @staticmethod
    def surface_density_profile(positions, nbins=50, rmax=None):
        r = np.hypot(positions[:, 0], positions[:, 1])
        if rmax is None:
            rmax = np.percentile(r, 99.0)

        edges = np.linspace(0, rmax, nbins + 1)
        r_mid = 0.5 * (edges[1:] + edges[:-1])
        sigma = np.zeros(nbins)

        for i in range(nbins):
            r_in, r_out = edges[i], edges[i + 1]
            area = np.pi * (r_out**2 - r_in**2)
            sigma[i] = np.count_nonzero((r >= r_in) & (r < r_out)) / area
        return r_mid, sigma

    @staticmethod
    def sersic_function(r, I0, re, n):
        b = 2.0 * n - 1.0 / 3.0
        return I0 * np.exp(-b * ((r / re)**(1 / n) - 1.0))

    @staticmethod
    def exponential_function(r, I0, rd):
        return I0 * np.exp(-r / rd)

    def fit_sersic(self, r, sigma, guess=(1.0, 1.0, 1.0), use_exponential=False):
        mask = sigma > 0
        r_fit, y_fit = r[mask], sigma[mask]
        if r_fit.size < 3:
            return None, None
        try:
            if use_exponential:
                return curve_fit(self.exponential_function, r_fit, y_fit, p0=guess[:2])
            else:
                return curve_fit(self.sersic_function, r_fit, y_fit, p0=guess)
        except Exception:
            return None, None

    def run(self):
        com_df = pd.read_csv(self.com_file)
        m33_df = com_df.query("(galaxy == 'M33') & (ptype == 2)").copy()
        if 'snap_int' not in m33_df.columns:
            m33_df['snap_int'] = m33_df['snapshot'].astype(int)

        records = []
        for snap in sorted(m33_df.snap_int.unique()):
            row = m33_df.loc[m33_df.snap_int == snap].squeeze()
            if row.empty:
                continue

            xcom, ycom, zcom, time_myr = row[['xcom', 'ycom', 'zcom', 'time_Myr']]
            fname = os.path.join(self.data_dir, f"M33_{snap:03d}.txt")
            if not os.path.exists(fname):
                continue

            _, _, data = Read(fname)
            star_idx = np.where((data['type'] == 2) | (data['type'] == 3))[0]

            pos = np.column_stack((data['x'][star_idx] - xcom,
                                   data['y'][star_idx] - ycom,
                                   data['z'][star_idx] - zcom))
            vel = np.column_stack((data['vx'][star_idx],
                                   data['vy'][star_idx],
                                   data['vz'][star_idx]))

            pos_face, _ = self.rotate_to_face_on(pos, vel)

            r_mid, sigma = self.surface_density_profile(pos_face, nbins=40, rmax=20.0)
            popt_exp, _ = self.fit_sersic(r_mid, sigma,
                                          guess=(sigma.max(), 2.0),
                                          use_exponential=True)
            popt_ser, _ = self.fit_sersic(r_mid, sigma,
                                          guess=(sigma.max(), 2.0, 1.0),
                                          use_exponential=False)

            records.append({
                "snapshot": snap,
                "time_Myr": time_myr,
                "exp_I0": popt_exp[0] if popt_exp is not None else 0.0,
                "exp_r_d": popt_exp[1] if popt_exp is not None else 0.0,
                "sersic_I0": popt_ser[0] if popt_ser is not None else 0.0,
                "sersic_re": popt_ser[1] if popt_ser is not None else 0.0,
                "sersic_n": popt_ser[2] if popt_ser is not None else 0.0
            })

        pd.DataFrame(records).to_csv(self.output_file, index=False)
        print(f"Created {self.output_file} with {len(records)} rows.")
