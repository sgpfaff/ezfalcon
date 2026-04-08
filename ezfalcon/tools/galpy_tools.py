import numpy as np
from ..util._galpy_bridge import _check_physical
import astropy.units as u
from galpy import df, potential

# --- Interface tools ----------------------------------------------------------------

def galpy_orbit_to_ezfalcon(orb):
    '''Convert a galpy orbit object to ezfalcon compatible pos and vel arrays.
    
     Parameters
     ----------
     orb : galpy.orbit.Orbit
         A galpy orbit object. You must turn physical on before passing it in, e.g
         with `orb.turn_physical_on()`.
         
     Returns
     -------
     pos : (N, 3) array
         Cartesian positions of sampled particles.
         Units: kpc
     vel : (N, 3) array
         Cartesian velocities of sampled particles.
         Units: kpc/Myr
     '''
    _check_physical(orb)
    pos = np.array([orb.x(return_physical=True), 
                    orb.y(return_physical=True), 
                    orb.z(return_physical=True)]) # kpc
    vel = (np.array([orb.vx(return_physical=True), 
                    orb.vy(return_physical=True), 
                    orb.vz(return_physical=True)])*u.km/u.s).to(u.kpc/u.Myr).value
    return pos.T, vel.T

# --- Sampling tools ----------------------------------------------------------------

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

def galpydfsampler(df, n, m_total, rmin=0.0, center_pos=[0, 0, 0], 
                   center_vel=[0, 0, 0], sampler_kwargs={}):
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
    rmin : float, Quantity, optional
        Minimum radius at which to sample. Default is 0.

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
    o = df.sample(n=n, rmin=rmin, return_orbit=True, **sampler_kwargs)
    pos, vel = galpy_orbit_to_ezfalcon(o)
    pos += np.asarray(center_pos)[:,None].T
    vel += np.asarray(center_vel)[:,None].T
    return pos, vel, np.repeat(m_total / n, n)

def galpysampler(pot, n, m_total, rmin=0.0, 
                 center_pos=[0, 0, 0], center_vel=[0, 0, 0],
                 sampler_kwargs={}):
    '''
    Sample from a galpy potential and return ezfalcon compatible 
    positions and velocities. Only supports spherical potentials.

    Parameters
    ----------
    pot : galpy.potential
        A galpy potential object. You 
        must turn physical on before passing it in, e.g. with `pot.turn_physical_on()`.

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
    _check_physical(pot)
    if isinstance(pot, potential.PlummerPotential):
        _df = df.isotropicPlummerdf(pot=pot)
    elif isinstance(pot, potential.HernquistPotential):
        _df = df.isotropicHernquistdf(pot=pot)
    elif isinstance(pot, potential.NFWPotential):
        _df = df.isotropicNFWdf(pot=pot)
    else:
        _df = df.eddingtondf(pot=pot)
    return galpydfsampler(_df, n=n, m_total=m_total, rmin=rmin, center_pos=center_pos, 
                            center_vel=center_vel, sampler_kwargs=sampler_kwargs)


def mkPlummer_galpy(m, b, n, center_pos=[0, 0, 0], center_vel=[0, 0, 0]):
    '''
    Generate the positions, velocities, and masses of 
    a Plummer sphere using galpy.

    Parameters
    ----------
    m : float
        Total mass of the Plummer sphere.
        Units: Msun
    b : float
        Scale radius of the Plummer sphere.
        Units: kpc
    n : int
        Number of particles to sample.
    rmin : float, Quantity, optional
        Minimum radius at which to sample. Default is 0.
    center_pos : array-like, optional
        Position of the center of the Plummer sphere.
        Units: kpc
    center_vel : array-like, optional
        Velocity of the center of the Plummer sphere.
        Units: kpc/Myr
    '''
    pot = potential.PlummerPotential(amp=m*u.Msun, b=b*u.kpc)
    return galpysampler(pot, n, m, center_pos=center_pos, center_vel=center_vel)


def mkKing_galpy(m:float, n, W0:float, rt=None, npts=None, rmin=0.0,
                 center_pos=[0, 0, 0], center_vel=[0, 0, 0]):
    '''
    Generate the positions, velocities, and masses of
    a King sphere using galpy.

    Parameters
    ----------
    m : float
        Total mass of the King sphere.
        Units: Msun
    n : int
        Number of particles to sample.
    W0 : float
        Dimensionless central potential 
        :math:`W_0 = \\Psi(0)/\\sigma^2` (in practice, needs to be :math:`\\lesssim 200`, where the DF is essentially isothermal).
    rt : float or Quantity, optional
        Tidal radius.
    npts : int
        Number of points to use to solve for :math:`\\Psi(r)`.
    rmin : float, Quantity, optional
            Minimum radius at which to sample. Default is 0.
    center_pos : array-like, optional
        Position of the center of the King sphere.
        Units: kpc
    center_vel : array-like, optional
        Velocity of the center of the King sphere.
        Units: kpc/Myr

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
    df_kwargs = {}
    if rt is not None:
        df_kwargs['rt'] = rt
    if npts is not None:
        df_kwargs['npt'] = npts
    sat_df = df.kingdf(W0=W0, M=m*u.Msun, **df_kwargs)
    return galpydfsampler(sat_df, n, m, rmin=rmin, 
                          center_pos=center_pos, center_vel=center_vel)


def mkNFW_galpy(m, n, rmin=0.0, center_pos=[0, 0, 0], center_vel=[0, 0, 0],
                **nfw_kwargs):
    '''
    Generate the positions, velocities, and masses of
    a NFW sphere using galpy.

    Parameters
    ----------
    m : float
        Total mass of the NFW sphere.
        Units: Msun
    n : int
        Number of particles to sample.
    rmin : float, Quantity, optional
        Minimum radius at which to sample. Default is 0.
    center_pos : array-like, optional
        Position of the center of the NFW sphere.
        Units: kpc
    center_vel : array-like, optional
        Velocity of the center of the NFW sphere.
        Units: kpc/Myr
    nfw_kwargs : keyword arguments to pass to the galpy isotropicNFWdf sampler.
         See galpy.df.isotropicNFWdf for details. Relevant kwargs include:
            - widrow (bool, optional):
                If True, use the approximate form from Widrow (2000), otherwise use improved fit that has <~1e-5 relative density errors
            - rmax (float or Quantity, optional):
                Maximum radius to consider; set to numpy.inf to evaluate NFW w/o cut-off
    
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
    pot = potential.NFWPotential(**nfw_kwargs)
    return galpysampler(pot, n, m, rmin=rmin, 
                        center_pos=center_pos, center_vel=center_vel,
                        sampler_kwargs={'widrow': nfw_kwargs.get('widrow', False), 'rmax': nfw_kwargs.get('rmax', np.inf)})
