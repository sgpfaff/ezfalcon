import numpy as np


def galpydfsampler(df, n, m_total, center_pos=[0, 0, 0], center_vel=[0, 0, 0], vo=220, ro=8.0):
    '''
    Sample from a galpy distribution function
    and return ezfalcon compatible positions
    and velocities.

    Parameters
    ----------
    df : galpy.df
        A galpy distribution function object.
    n : int
        Number of particles to sample.
    m_total : float
        Total mass of the sampled component.
        Units: Msun

    Returns
    -------
    pos : (N, 3) array
        Cartesian positions of sampled particles.
        Units: kpc
    vel : (N, 3) array
        Cartesian velocities of sampled particles.
        Units: kpc/Myr
    masses : (N,) array
        Masses of sampled particles.
    '''
    df.turn_physical_on()
    o = df.sample(n=n, return_orbit=True)
    # Orbit methods with physical on return kpc and km/s
    pos = np.array([o.x(return_physical=True, vo=vo, ro=ro), 
                    o.y(return_physical=True, vo=vo, ro=ro), 
                    o.z(return_physical=True, vo=vo, ro=ro)])
    vel = np.array([o.vx(return_physical=True, vo=vo, ro=ro), 
                    o.vy(return_physical=True, vo=vo, ro=ro), 
                    o.vz(return_physical=True, vo=vo, ro=ro)]) * FROM_GALPY_TO_INTERNAL['vel']
    
    pos += np.asarray(center_pos)[:,None]
    vel += np.asarray(center_vel)[:,None]
    return pos.T, vel.T, np.repeat(m_total / n, n)