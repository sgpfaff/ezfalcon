import numpy as np
from ..util._galpy_bridge import _check_physical
import astropy.units as u
from galpy import df

SUPPORTED_GALPY_DFS = (
    df.isotropicHernquistdf,
    df.isotropicPlummerdf,
    df.isotropicNFWdf,
    df.kingdf,
    df.eddingtondf
)

def _check_df(df):
    '''Currently only spherical galpy dfs are supported.'''
    if not isinstance(df, SUPPORTED_GALPY_DFS):
        raise ValueError(f"Unsupported galpy df type: {type(df)}. Supported types: {SUPPORTED_GALPY_DFS}")

def galpydfsampler(df, n, m_total, center_pos=[0, 0, 0], center_vel=[0, 0, 0], vo=220, ro=8.0):
    '''
    Sample from a galpy spherical distribution 
    function and return ezfalcon compatible 
    positions and velocities.

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
    _check_physical(df)
    _check_df(df)
    o = df.sample(n=n, return_orbit=True)
    pos, vel = galpy_orbit_to_ezfalcon(o)
    pos += np.asarray(center_pos)[:,None].T
    vel += np.asarray(center_vel)[:,None].T
    return pos, vel, np.repeat(m_total / n, n)


def galpy_orbit_to_ezfalcon(o):
    _check_physical(o)
    pos = np.array([o.x(return_physical=True), 
                    o.y(return_physical=True), 
                    o.z(return_physical=True)]) # kpc
    vel = (np.array([o.vx(return_physical=True), 
                    o.vy(return_physical=True), 
                    o.vz(return_physical=True)])*u.km/u.s).to(u.kpc/u.Myr).value
    return pos.T, vel.T
    