from galpy.potential import evaluatePotentials, evaluateDensities
import numpy as np



def natural_radius(r_kpc, ro=8.0):
    """
    Convert physical radius in kpc to galpy natural units (R = r_kpc / ro).
    """
    return r_kpc / ro

def natural_mass(M_solar, ro=8.0, vo=220.0):
    """
    Convert physical mass in solar masses to galpy natural units (M = M_solar * G / (ro * vo²)).
    """
    from astropy.constants import G
    from astropy import units as u
    M_nat = (M_solar * u.Msun * G / (ro * u.kpc * vo**2 * u.km**2 / u.s**2)).decompose()
    return M_nat.value


class HaloModel:
    """
    Spherically symmetric galactic halo model pairing a galpy potential with a
    matched isotropic distribution function.

    All inputs and outputs use galpy internal (natural) units:
      - length:  units of ro  (default 8 kpc,   so R=1 means 8 kpc)
      - energy:  units of vo² (default 220 km/s, so E=-1 means -48400 km²/s²)
      - density: vo² / (4π G ro²)
      - DF:      vo² / (4π G ro²) / vo³  =  1 / (4π G ro² vo)

    Parameters
    ----------
    pot : galpy potential instance
        A spherically symmetric galpy potential (e.g. HernquistPotential).
    df_class : type
        An uninstantiated galpy DF class (e.g. isotropicHernquistdf). It will
        be instantiated internally as df_class(pot=pot, ro=ro, vo=vo, **df_kwargs).
    ro : float
        Distance scale in kpc. Default 8.0.
    vo : float
        Velocity scale in km/s. Default 220.0.
    **df_kwargs
        Additional keyword arguments forwarded to the DF constructor.
    """

    def __init__(self, pot, df_class, ro=8.0, vo=220.0, **df_kwargs):
        self._ro = ro
        self._vo = vo
        self._pot = pot
        self._df = df_class(pot=pot, ro=ro, vo=vo, **df_kwargs)

    def density(self, R):
        """
        Density at spherical radius R (galpy units, R=1 means ro kpc).

        Returns density in galpy natural units.
        """
        return evaluateDensities(self._pot, R, 0.0, use_physical=False)

    def potential(self, R):
        """
        Gravitational potential at spherical radius R (galpy units).

        Galpy convention: Phi(inf) = 0; bound orbits have Phi < 0.
        Returns potential in galpy natural units (units of vo²).
        """
        return evaluatePotentials(self._pot, R, 0.0, use_physical=False)
    
    def mass_enclosed(self, R):
        """Mass enclosed within radius R (galpy units)
        Returns mass in internal units.
        """
        return self._pot.mass(R, use_physical=False)

    def df(self, E):
        """
        Isotropic distribution function f(E) at dimensionless energy E (units of vo²).

        Returns DF in galpy natural units.
        """
        # fE requires a numpy array (calls .shape on its input); squeeze back to scalar
        result = self._df.fE(np.atleast_1d(E))
        return result.squeeze()[()]


# ##TEST
# from galpy.potential import HernquistPotential
# from galpy.df import isotropicHernquistdf
# from galpy.util import conversion
# import astropy.units as u
# from astropy.constants import G
# import matplotlib.pyplot as plt
# ro,vo = 8.0, 220.0
# r_array = np.linspace(0.1,30,100) #kpc
# r_nat = r_array / ro
# print(G)
# M_nat = (1e12*u.Msun * G / (ro*u.kpc * vo**2*u.m**2/u.s**2)).decompose() #Msun

# hq_pot = HernquistPotential(amp=2*M_nat.value, a=20/ro, ro=ro, vo=vo)
# hm = HaloModel(pot=hq_pot, df_class=isotropicHernquistdf, ro=ro, vo=vo)

# plt.plot(r_array, hm.density(r_nat)*conversion.dens_in_msolpc3(ro, vo), label='density')
# plt.loglog()
# plt.show()