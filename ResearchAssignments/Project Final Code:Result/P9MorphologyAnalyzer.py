import numpy as np
import pandas as pd
import os
from pathlib import Path
from P1Read import Read
from P2CenterOfMass import CenterOfMass

class MorphologyAnalyzer:
    def __init__(self, datapath='M33/', output_file='M33_morphology_results.txt'):
        self.datapath = Path(datapath)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def rotate_frame(x, y, z, L):
        Lhat = L / np.linalg.norm(L)
        zhat = np.array([0.0, 0.0, 1.0])
        v = np.cross(Lhat, zhat)
        s = np.linalg.norm(v)
        c = np.dot(Lhat, zhat)
        if s == 0:
            return x, y, z

        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / s**2)

        xyz_rot = R @ np.vstack((x, y, z))
        return xyz_rot[0], xyz_rot[1], xyz_rot[2]

    @staticmethod
    def compute_angular_momentum(x, y, z, vx, vy, vz, m):
        Lx = np.sum(m * (y * vz - z * vy))
        Ly = np.sum(m * (z * vx - x * vz))
        Lz = np.sum(m * (x * vy - y * vx))
        return np.array([Lx, Ly, Lz])

    @staticmethod
    def angle_between_vectors(a, b):
        cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.arccos(cosang)

    def measure_warp_and_scaleheight(self, x, y, z, vx, vy, vz, m, nbins=10):
        L_global = self.compute_angular_momentum(x, y, z, vx, vy, vz, m)
        x_r, y_r, z_r = self.rotate_frame(x, y, z, L_global)
        vx_r, vy_r, vz_r = self.rotate_frame(vx, vy, vz, L_global)

        r = np.hypot(x_r, y_r)
        rmax = r.max()
        edges = np.linspace(0, rmax, nbins + 1)
        mid = 0.5 * (edges[:-1] + edges[1:])

        warp_angle = np.full(nbins, np.nan)
        scale_height = np.full(nbins, np.nan)

        for i, (rin, rout) in enumerate(zip(edges[:-1], edges[1:])):
            idx = (r >= rin) & (r < rout)
            if idx.sum() < 10:
                continue

            L_local = self.compute_angular_momentum(x_r[idx], y_r[idx], z_r[idx],
                                                    vx_r[idx], vy_r[idx], vz_r[idx],
                                                    m[idx])
            warp_angle[i] = np.degrees(self.angle_between_vectors(L_local, L_global))
            scale_height[i] = np.std(z_r[idx])
        return mid, warp_angle, scale_height

    def analyze_snapshots(self, snaprange=range(0, 802, 50)):
        with open(self.output_file, 'w') as f:
            f.write("# snap  time[Myr]  R_bin(kpc)  warp[deg]  h_z[kpc]\n")

            for snap in snaprange:
                fname = self.datapath / f"M33_{snap:03d}.txt"
                if not fname.exists():
                    continue

                time_val, _, data = Read(str(fname))
                com = CenterOfMass(data, ptype=2)
                xcom, ycom, zcom = com.COM_P(delta=0.1)
                vxcom, vycom, vzcom = com.COM_V(xcom, ycom, zcom, rvmax=15.0)

                x = com.x - xcom
                y = com.y - ycom
                z = com.z - zcom
                vx = com.vx - vxcom
                vy = com.vy - vycom
                vz = com.vz - vzcom
                m = com.m

                mid, warp, h_z = self.measure_warp_and_scaleheight(x, y, z, vx, vy, vz, m, nbins=10)

                for R, w, h in zip(mid, warp, h_z):
                    f.write(f"{snap:03d}  {time_val:9.2f}  {R:6.2f}  {w:6.2f}  {h:7.3f}\n")

        print(f"Done. Results saved to {self.output_file}")
