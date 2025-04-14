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
    total = int(value2)  # total number of particles

    file.readline()
    file.readline()

    ptype_list = []
    m_list = []
    x_list = []
    y_list = []
    z_list = []
    vx_list = []
    vy_list = []
    vz_list = []

    for _ in range(total):
        line = file.readline()
        if not line:
            # in case of truncated file
            break

        vals = line.split()
        ptype_list.append(float(vals[0]))
        m_list.append(float(vals[1]))
        x_list.append(float(vals[2]))
        y_list.append(float(vals[3]))
        z_list.append(float(vals[4]))
        vx_list.append(float(vals[5]))
        vy_list.append(float(vals[6]))
        vz_list.append(float(vals[7]))

    file.close()

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

    data_array = np.zeros(total, dtype=dt)

    data_array['type'] = ptype_list
    data_array['m']    = m_list
    data_array['x']    = x_list
    data_array['y']    = y_list
    data_array['z']    = z_list
    data_array['vx']   = vx_list
    data_array['vy']   = vy_list
    data_array['vz']   = vz_list

    return time, total, data_array
