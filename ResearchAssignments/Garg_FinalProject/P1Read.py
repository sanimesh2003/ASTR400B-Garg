# Authors of this code : Aniemsh Garg, Google Gemini
#
# Aniemsh Garg  - code structures and writing comments
# Google Gemini - used as a debugging tool,
#                 also used to create better commenting structure
#
# Labs and homeworks from the class ASTR400B taught by Dr Gurtina Besla,
# were a resource in sourcing and completion of my code

# This file is a derivative of the readfile used in class

import numpy as np

def Read(filename):
    """
    Function to read a GADGET-style snapshot data file and return:
      1) Time in Myr
      2) Total number of particles
      3) A structured numpy array with fields: [type, m, x, y, z, vx, vy, vz]
    PARAMETERS
    ----------
    filename : str
        Path to the data file, e.g. 'MW_000.txt'
    RETURNS
    -------
    time  : float
        The snapshot time in Myr
    total : int
        The total number of particles in the file
    data  : np.ndarray
        A structured array containing columns:
          'type', 'm', 'x', 'y', 'z', 'vx', 'vy', 'vz'
    """

    file = open(filename, 'r')

    line1 = file.readline()
    label, value = line1.split()
    time = float(value)  # Myr

    line2 = file.readline()
    label2, value2 = line2.split()
    expected_total_particles = int(value2)  # total number of particles (from header)

    file.readline()
    file.readline()

    ptype_arr = np.zeros(expected_total_particles, dtype=float)
    m_arr     = np.zeros(expected_total_particles, dtype=float)
    x_arr     = np.zeros(expected_total_particles, dtype=float)
    y_arr     = np.zeros(expected_total_particles, dtype=float)
    z_arr     = np.zeros(expected_total_particles, dtype=float)
    vx_arr    = np.zeros(expected_total_particles, dtype=float)
    vy_arr    = np.zeros(expected_total_particles, dtype=float)
    vz_arr    = np.zeros(expected_total_particles, dtype=float)

    actual_particles_read = 0
    for i in range(expected_total_particles):
        line = file.readline()
        if not line:
            # in case of truncated file
            break

        vals = line.split()
        # Basic check for sufficient values, can be made more robust
        if len(vals) == 8:
            try:
                ptype_arr[actual_particles_read] = float(vals[0])
                m_arr[actual_particles_read]     = float(vals[1])
                x_arr[actual_particles_read]     = float(vals[2])
                y_arr[actual_particles_read]     = float(vals[3])
                z_arr[actual_particles_read]     = float(vals[4])
                vx_arr[actual_particles_read]    = float(vals[5])
                vy_arr[actual_particles_read]    = float(vals[6])
                vz_arr[actual_particles_read]    = float(vals[7])
                actual_particles_read += 1
            except ValueError:
                continue # Skip this particle
        else:
            continue # Skip this particle

    file.close()

    if actual_particles_read < expected_total_particles:
        ptype_arr = ptype_arr[:actual_particles_read]
        m_arr     = m_arr[:actual_particles_read]
        x_arr     = x_arr[:actual_particles_read]
        y_arr     = y_arr[:actual_particles_read]
        z_arr     = z_arr[:actual_particles_read]
        vx_arr    = vx_arr[:actual_particles_read]
        vy_arr    = vy_arr[:actual_particles_read]
        vz_arr    = vz_arr[:actual_particles_read]

    dt = np.dtype([
        ('type', float),
        ('m', float),
        ('x', float),
        ('y', float),
        ('z', float),
        ('vx', float),
        ('vy', float),
        ('vz', float)
    ])

    data_array = np.zeros(actual_particles_read, dtype=dt)

    data_array['type'] = ptype_arr
    data_array['m']    = m_arr
    data_array['x']    = x_arr
    data_array['y']    = y_arr
    data_array['z']    = z_arr
    data_array['vx']   = vx_arr
    data_array['vy']   = vy_arr
    data_array['vz']   = vz_arr

    total = actual_particles_read
    return time, total, data_array
