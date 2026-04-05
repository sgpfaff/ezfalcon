"""
Simulation class for ezfalcon.
"""

import numpy as np
from .component import Component
from ..dynamics import _integrate, self_gravity
import galpy
from ..util._galpy_bridge import _galpy_pot_to_acc_fn, _galpy_pot_to_pot_fn, _check_physical, _check_supported_pot
import functools
import warnings

_USE_CACHED_DEFAULT = object()  # sentinel for "caller didn't pass use_cached"

def _resolve_use_cached(func):
    '''
    Decorator that resolves *use_cached* dynamically:
      - If the caller didn't pass use_cached:
          * method given  -> use_cached = False  (compute on-the-fly)
          * method absent -> use_cached = _has_run (cache if available, else error)
      - If the caller explicitly passed use_cached=True:
          * before run()       -> error (no cache exists)
          * with method given   -> error (conflicting intent)
      - If the caller explicitly passed use_cached=False:
          * method absent  -> error (need a method to compute)
    '''
    @functools.wraps(func)
    def wrapper(*args, use_cached=_USE_CACHED_DEFAULT, method=None, **kwargs):
        sim = args[0]
        explicit = use_cached is not _USE_CACHED_DEFAULT

        if not explicit:
            if method is not None:
                use_cached = False
            else:
                use_cached = sim._has_run
        else:
            if use_cached and not sim._has_run:
                raise ValueError("Cannot use cached results before run(). "
                    "Please set use_cached to False and provide a method "
                    "for computing self-gravity.")
            if use_cached and method is not None:
                raise ValueError("`method` should not be specified if "
                    "`use_cached` is True, since the cached self-gravity "
                    "was computed using a specific method. Please set "
                    "`use_cached` to False to specify a method for "
                    "computing self-gravity.")

        if not use_cached and method is None:
            if not explicit and not sim._has_run:
                raise ValueError("No cached results available — the simulation "
                    "has not been run yet. Please call run() first, or provide "
                    "a method (e.g. method='direct') to compute on-the-fly.")
            raise ValueError("`use_cached` is False but no `method` was provided. "
                "Please specify a method (e.g. method='direct') to compute "
                "self-gravity, or set `use_cached` to True.")

        return func(*args, use_cached=use_cached, method=method, **kwargs)
    return wrapper


def _resolve_t(func):
    '''
    Decorator that resolves *t* based on *use_cached*:
      - use_cached=True  -> t can be ... (all snapshots) or int/float
      - use_cached=False -> t must be a single snapshot (... is rejected)

    Must be applied AFTER @_resolve_use_cached (i.e. listed BEFORE it
    in stacked decorator order).
    '''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sim = args[0]
        use_cached = kwargs.get('use_cached', True)

        # Extract t from positional args or kwargs
        if len(args) > 1:
            t = args[1]
            args = (args[0],) + args[2:]
        else:
            t = kwargs.pop('t', ...)

        if use_cached:
            t = sim._ti(t, vectorized=True)
        else:
            if t is ...:
                raise TypeError(
                    "Cannot compute on-the-fly for all times. "
                    "Please provide an integer index or a float time for t. "
                    "You will have to manually loop over snapshots.")
            t = sim._ti(t, vectorized=False)

        return func(*args, t=t, **kwargs)
    return wrapper


class Sim:
    """
    Self-gravitating N-body simulation.
    """

    def __init__(self):
        self._host = None

        # Particle arrays -- built incrementally by add_particles()
        self._init_pos = None     # (N, 3) kpc
        self._init_vel = None     # (N, 3) kpc/Myr
        self._mass = None         # (N,) Msun
        self._slices = {}         # component name -> slice

        # Snapshot arrays -- populated by run()
        self._positions = None       # (n_snap, N, 3) kpc
        self._velocities = None      # (n_snap, N, 3) kpc/Myr
        self._cached_self_acc = None # (n_snap, N, 3) kpc/Myr^2
        # self._cached_ext_acc = None  # (n_snap, N, 3) kpc/Myr^2
        self._cached_self_pot = None # (n_snap, N)    kpc^2/Myr^2
  
        self._times = None           # (n_snap,) Myr
        self._has_run = False
        self._self_gravity_on = True
        self._ext_acc_fns = []     # list of functions that take (pos, t) and return (N, 3) acc
        self._ext_pot_fns = []

    def _ti(self, t, vectorized=True):
        """
        Resolve *t* to a snapshot index.

        int -> direct index, float -> nearest time.
        """
        if isinstance(t, (int, np.integer)) or t == ...:
            if t == ...:
                if vectorized:
                    return t
                else:
                    raise TypeError("This method is not vectorized, so t cannot be a list or ellipse. Please provide" \
                    " an integer index or a float time.")
            else:
                if t > len(self._times) - 1 or t < -len(self._times):
                    print(f"Time index {t} is out of bounds for simulation with {len(self._times)} snapshots. Please provide an index within [-{len(self._times)}, {len(self._times)-1}].")
                    raise IndexError(f"Time index {t} is out of bounds for simulation with {len(self._times)} snapshots. Please provide an index within [-{len(self._times)}, {len(self._times)-1}].")
                else:
                    return int(t)
        else:
            if not isinstance(t, (float, np.floating)):
                raise TypeError("t must be an int index, a float time, or ellipsis.")
            else:
                if t < self._times[0] or t > self._times[-1]:
                    raise ValueError(f"t={t} Myr is out of bounds for simulation time range [{self._times[0]}, {self._times[-1]}] Myr.")
                else:
                    return int(np.argmin(np.abs(self._times - t)))

    def __getattr__(self, name):
        '''
        Access components as attributes, e.g. sim.sat.pos(t=10).
        '''
        if name.startswith("_"):
            raise AttributeError(name)
        slices = self.__dict__.get("_slices", {})
        if name in slices:
            return Component(self, slices[name], name)
        raise AttributeError(
            f"\'{type(self).__name__}\' has no attribute or component named {name!r}"
        )
    
    # --- Setup ---------------------------------------------------------------------------------                                                   

    def turn_self_gravity_on(self):
        '''
        Turn self-gravity on for the simulation. 
        This is on by default.
        '''
        self._self_gravity_on = True

    def turn_self_gravity_off(self):
        '''
        Turn self-gravity off for the simulation.

        Methods the acceleration and potential due
        to self-gravity will be zero.
        '''
        self._self_gravity_on = False

    def add_particles(self, name, pos, vel, mass):
        """
        Add a named particle component.
        Provide (pos, vel, mass) directly [kpc, kpc/Myr, Msun].

        Parameters
        ----------
        name : str
            Name of the component, e.g. 'sat' or 'host'.
        pos : (N, 3) array
            Initial positions of particles.
            Units: `kpc`
        vel : (N, 3) array
            Initial velocities of particles.
            Units: `kpc/Myr`
        mass : (N,) array
            Masses of particles.
            Units: `Msun`

        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If the component already exists or if the input arrays have incompatible shapes.
        TypeError
            If the input types are incorrect.
        RuntimeError
            If the simulation has already been run.
        """
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)
        mass = np.asarray(mass, dtype=np.float64)
        if self._has_run:
            raise RuntimeError("Cannot add components after run()")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if name in self._slices:
            raise ValueError(f"Component \'{name}\' already exists.")
        if pos.ndim != 2  or pos.shape[1] != 3:
            raise ValueError(f"pos must be shape (N, 3), received {pos.shape}")
        if vel.ndim != 2  or vel.shape[1] != 3:
                raise ValueError(f"vel must be shape (N, 3), received {vel.shape}")
        if mass.ndim != 1:
            raise ValueError(f"mass must be shape (N,), received {mass.shape}")
        if not (pos.shape[0] == vel.shape[0] == mass.shape[0]):
            raise ValueError(f"pos, vel, mass must have same number of particles, received {pos.shape[0]}, {vel.shape[0]}, {mass.shape[0]}.")

        # Build slice and append to flat arrays
        n = pos.shape[0]
        offset = 0 if self._mass is None else self._mass.shape[0]
        self._slices[name] = slice(offset, offset + n)

        if self._mass is None:
            self._init_pos = pos
            self._init_vel = vel
            self._mass = mass
        else:
            self._init_pos = np.concatenate([self._init_pos, pos])
            self._init_vel = np.concatenate([self._init_vel, vel])
            self._mass = np.concatenate([self._mass, mass])

        # Build single-snapshot arrays so accessors work immediately
        N = self._mass.shape[0]
        self._positions = self._init_pos.reshape(1, N, 3)
        self._velocities = self._init_vel.reshape(1, N, 3)
        self._times = np.array([0.0])
    
    def add_external_pot(self, pot):
        '''
        Add an external potential to the simulation.

        Parameters
        ----------
        pot : galpy.potential.Potential
            External potential to add.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the potential is not a galpy Potential object.
        
        Warnings
        --------
        UserWarning
            If the provided galpy potential has physical outputs turned off. 
            In this case
        
        '''
        if isinstance(pot, galpy.potential.Potential):
            _check_supported_pot(pot)
            _check_physical(pot)
            self._ext_pot_fns.append(_galpy_pot_to_pot_fn(pot))
            self._ext_acc_fns.append(_galpy_pot_to_acc_fn(pot))
        else:
            raise TypeError("External potential must be a galpy Potential object.")

    def run(self, t_end, dt, dt_out, method='falcON', 
            cache_self_gravity=True, cache_self_potential=True, **kwargs):
        """
        Run the simulation to *t_end* [Myr].

        Parameters
        ----------
        t_end : float
            End time of the simulation.
            Units: Myr
        dt : float
            Timestep for integration.
            Units: Myr
        dt_out : float
            Output interval.
            Units: Myr
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON' (default): fast multipole method implemented in falcON.
            - 'direct': direct summation.
        cache_self_gravity : bool, optional
            Whether to cache the self-gravity acceleration at each output snapshot. Default is True.
        cache_self_potential : bool, optional
            Whether to cache the self-gravitational potential at each output snapshot. Default is True.
        cache_ext_acc : bool, optional
            Whether to cache the external acceleration at each output snapshot. Default is True.
        cache_ext_pot : bool, optional
            Whether to cache the external potential at each output snapshot. Default is True.
        **kwargs 
            Additional keyword arguments to pass to the gravity method. 

            For 'falcON', these include:
            
            - eps ((N,) array or scalar): Gravitational softening length (kpc)
            - theta (float, optional): Tree opening angle. Default is 0.6. Smaller = more accurate but slower.
            - kernel (int, optional): Softening kernel: 0=Plummer, 1=default (~r^-7), 2,3=faster decay.

            For 'direct', these include:
            - eps (scalar): Gravitational softening length (kpc)

            For 'direct_C', these include:
            - eps: Gravitational softening length (kpc)
        """
        if dt <= 0 or dt_out <= 0 or t_end <= 0:
            raise ValueError("dt, dt_out, and t_end must be positive.")
        if dt_out < dt:
            raise ValueError("dt_out must be greater than or equal to dt.")
        
        (self._positions, self._velocities, self._times,
         self._cached_self_acc, self._cached_self_pot) = _integrate(
                    pos = self._init_pos, 
                    vel = self._init_vel, 
                    mass = self._mass,
                    include_self_gravity = self._self_gravity_on, 
                    self_gravity_method=method,
                    extra_acc = self._ext_acc_fns,
                    t_end = t_end,
                    dt = dt,
                    dt_out = dt_out,
                    return_self_potential = cache_self_potential,
                    return_self_gravity = cache_self_gravity,
                    **kwargs
                )
        self._has_run = True

    # --- Position Accessors -----------------------------------------------------------------

    def pos(self, t=...):
        '''
        Particle positions (x,y,z) at *t*.

        Units: `kpc`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        pos : (len(t), n_particles, 3) array or (n_particles, 3) array
            Positions at *t*.
            Units: `kpc`
        '''
        return self._positions[self._ti(t)]

    def x(self, t=...):
        '''
        Particle x-positions of all particles at *t*.

        Units: `kpc`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        x : (len(t), n_particles) array or (n_particles,) array
            x-positions at *t*.
            Units: `kpc`
        '''
        return self._positions[self._ti(t), :, 0]
 
    def y(self, t=...):
        '''
        Particle y-positions of all particles at *t*.

        Units: `kpc`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        y : (len(t), n_particles) array or (n_particles,) array
            y-positions at *t*.
            Units: `kpc`
        '''
        return self._positions[self._ti(t), :, 1]

    def z(self, t=...):
        '''
        Particle z-positions of all particles at *t*.

        Units: `kpc`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.

        Returns
        -------
        z : (len(t), n_particles) array or (n_particles,) array
            z-positions at *t*.
            Units: `kpc`
        '''
        return self._positions[self._ti(t), :, 2]
    
    # --- Velocity Accessors -----------------------------------------------------------------

    def vel(self, t=...):
        '''
        Particle velocities (vx,vy,vz) at *t*.

        Units: `kpc / Myr`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        vel : (len(t), n_particles, 3) array or (n_particles, 3) array
            Velocities at *t*.
            Units: `kpc / Myr`
        '''
        return self._velocities[self._ti(t)]

    def vx(self, t=...):
        '''
        x-component of particle velocities at *t*.

        Units: :math:`kpc / Myr`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        vx : (len(t), n_particles) array or (n_particles,) array
            x-component of velocities at *t*.
            Units: :math:`kpc / Myr`
        '''
        return self._velocities[self._ti(t), :, 0]

    def vy(self, t=...):
        '''
        y-component of particle velocities at *t*.

        Units: :math:`kpc / Myr`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        vy : (len(t), n_particles) array or (n_particles,) array
            y-component of velocities at *t*.
            Units: :math:`kpc / Myr`
        '''
        return self._velocities[self._ti(t), :, 1]

    def vz(self, t=...):
        '''
        z-component of particle velocities at *t*.

        Units: `kpc / Myr`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        vz : (len(t), n_particles) array or (n_particles,) array
            z-component of velocities at *t*.

            Units: `kpc / Myr`
        '''
        return self._velocities[self._ti(t), :, 2]
    
    # --- Energy Accessors -----------------------------------------------------------------

    # --- Potential Energy --- #
    def compute_external_pot(self, t=...):
        '''
        External potential of particles at *t*.

        Units: `kpc^2 / Myr^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        ext_pot : (len(t), n_particles) array
            External potential at each snapshot.
            Units: `Msun kpc^2 / Myr^2`
        '''
        t = self._ti(t, vectorized=True)
        if isinstance(t, (int, np.integer)):
            ext_pot = np.zeros(self._mass.shape[0])
            for fn in self._ext_pot_fns:
                ext_pot += fn(self.pos(t=t), t=t)
        else:
            warnings.warn("Computing external potential on-the-fly for multiple snapshots may be slow.")
            if t is ...:
                t = self._times
            ext_pot = np.zeros((len(t), self._mass.shape[0]))
            for fn in self._ext_pot_fns:
                for i, t_i in enumerate(t):
                    ext_pot[i] += fn(self.pos(t=t_i), t=t_i)
        return self._mass * ext_pot
    
    @_resolve_use_cached
    @_resolve_t
    def self_potential(self, t=..., use_cached=True, method=None, **kwargs):
        '''
        Self-gravitational potential of particles at *t*.

        Units: `Msun kpc^2 / Myr^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        use_cached : bool, optional
            Whether to use cached self-potential if available. Default is True.
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON' (default): fast multipole method implemented in falcON.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method. 

            For 'falcON', these include:
            - eps: Gravitational softening length (kpc)
            - theta: Tree opening angle (default 0.6). Smaller = more accurate but slower.

            For 'direct', these include:
            - eps: Gravitational softening length (kpc)

        Returns
        -------
        self_pot : (len(t), n_particles) array
            Self-gravitational potential of each particle at each snapshot.

            Units: `Msun kpc^2 / Myr^2`
        '''
        if use_cached and self._cached_self_pot is not None:
            return self._mass * self._cached_self_pot[self._ti(t, vectorized=True)]
        elif use_cached and self._cached_self_pot is None:
            raise ValueError("Cached self-potential is not available. Please set use_cached to False and provide a method for computing self-gravity.")
        else:
            _, self_pot = self_gravity(self.pos(t=self._ti(t, vectorized=False)), self._mass, method=method, **kwargs)
            return self._mass * self_pot
    
    @_resolve_use_cached
    @_resolve_t
    def PE(self, t=..., use_cached=True, method=None, **kwargs):
        '''
        Total potential energy of particles at *t*.

        Units:  `Msun kpc^2 / Myr^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        use_cached : bool, optional
            Whether to use cached self-potential if available. Default is True.
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON' (default): fast multipole method implemented in falcON.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method. 

            For 'falcON', these include:
            - eps: Gravitational softening length (`kpc`)
            - theta: Tree opening angle (default 0.6). Smaller = more accurate but slower.

            For 'direct', these include:
            - eps: Gravitational softening length (`kpc`)

        Returns
        -------
        PE : (len(t), n_particles) array
            Total potential energy of each particle at each snapshot.
            Units: `Msun kpc^2 / Myr^2`
        '''
        return self.self_potential(t=t, method=method, use_cached=use_cached, **kwargs) + self.compute_external_pot(t=t)
    
    # --- Kinetic Energy --- #

    def KE(self, t=...):
        '''
        Kinetic energy of particles at *t*.

        Units:  `Msun kpc^2 / Myr^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        KE : (len(t), n_particles) array
            Kinetic energy of each particle at each snapshot.
            Units: `Msun kpc^2 / Myr^2`
        '''
        return 0.5 * self._mass * np.sum(self.vel(t=t) ** 2, axis=-1)

    # --- Total Energy --- #
    @_resolve_use_cached
    @_resolve_t
    def energy(self, t=..., use_cached=True, method=None, **kwargs):
        """
        Energy of particles at *t*.
        
        Units:  `Msun kpc^2 / Myr^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        use_cached : bool, optional
            Whether to use cached self-gravity from integration if available. Default is True.
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON' (default): fast multipole method implemented in falcON.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method. 

            For 'falcON', these include:
            - eps: Gravitational softening length (kpc)
            - theta: Tree opening angle (default 0.6). Smaller = more accurate but slower.

            For 'direct', these include:
            - eps: Gravitational softening length (kpc)
        
        Returns
        -------
        energy : (len(t), n_particles) array
            Total energy of each particle at each snapshot.
            Units: `Msun kpc^2 / Myr^2`
        """
        return self.KE(t=t) + self.PE(t=t, method=method, use_cached=use_cached, **kwargs)
    
    @_resolve_use_cached
    @_resolve_t
    def system_energy(self, t=..., use_cached=True, method=None,  **kwargs):
        r"""
        Total conserved system energy at *t*.
        
        Units: `Msun kpc^2 / Myr^2`

        .. math::

            E = \sum_i \frac{1}{2} m_i |v_i|^2 + \frac{1}{2} \sum_i m_i \Phi_{\text{self},i} + \sum_i m_i \Phi_{\text{ext},i}


        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        use_cached : bool, optional
            Whether to use cached self-potential from integration 
            if available. Default is True.
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON' (default): fast multipole method implemented in falcON.
            - 'direct': direct summation.
        
        **kwargs
            Additional keyword arguments to pass to the gravity method.

            For 'falcON', these include:
            - eps: Gravitational softening length (kpc)
            - theta: Tree opening angle (default 0.6). Smaller = more accurate but slower.

            For 'direct', these include:
            - eps: Gravitational softening length (kpc)

        Returns
        -------
        energy : float
            Total energy of the system at time t.
            Units: :math:`Msun kpc^2 / Myr^2`
        """
        return (np.sum(self.KE(t=t), axis=-1) + 
                0.5 * np.sum(self.self_potential(t=t, method=method, 
                                                 use_cached=use_cached, **kwargs), axis=-1) + 
                np.sum(self.compute_external_pot(t=t), axis=-1)
                )
    
    @_resolve_use_cached
    def dE(self, t=..., use_cached=True, method=None, **kwargs):
        '''
        Percent change in total energy over the simulation time.

        Parameters
        ----------
        method : str
            Method to use for computing self-gravity. Included options are:
            - 'falcON': fast multipole method implemented in falcON.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method.

            For 'falcON', these include:
            - eps: Gravitational softening length (kpc)
            - theta: Tree opening angle (default 0.6). Smaller = more accurate but slower.

            For 'direct', these include:
            - eps: Gravitational softening length (kpc)
        use_cached : bool, optional
            Whether to use cached self-potential from integration if available. Default is True.

        Returns
        -------
        dE : (n_snaps,) array
            Percent change in total energy at each snapshot.
        '''
        if use_cached:
            Es = self.system_energy(t=t, use_cached=use_cached, method=method, **kwargs)
            E0 = Es[0]
        else:
            if t is ...:
                Es = np.array([self.system_energy(t=t_i, use_cached=False, method=method, **kwargs) for t_i in self._times])
                E0 = Es[0]
            else:
                Es = self.system_energy(t=t, use_cached=False, method=method, **kwargs)
                E0 = self.system_energy(t=0, use_cached=False, method=method, **kwargs)
        return np.abs((Es - E0) / E0)

    # --- Acceleration Accessors -----------------------------------------------------------------

    # --- Self-Gravity --- #

    @_resolve_use_cached
    @_resolve_t
    def self_gravity(self, t=..., use_cached=True, method=None,  **kwargs):
        '''
        Compute the self-gravity acceleration of each 
        particle in the component at *t*.
        
        Units: `kpc / Myr^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        use_cached : bool, optional
            Whether to use cached self-gravity if available. Default is True.
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON' (default): fast multipole method implemented in falcON.
            - 'direct': direct summation.
        
        **kwargs
            Additional keyword arguments to pass to the gravity method.
        
        Returns
        -------
        self_acc : (n_snaps, N, 3) array
            Self-gravity acceleration of each particle at each snapshot.
            [ax, ay, az]
            Units: `kpc / Myr^2`
        '''
        if self._self_gravity_on:
            if use_cached and self._cached_self_acc is not None:
                return self._cached_self_acc[self._ti(t, vectorized=True)]
            elif use_cached and self._cached_self_acc is None:
                raise ValueError("Cached self-gravity is not available. Please set use_cached to False and provide a method for computing self-gravity.")
            else:
                return self_gravity(self.pos(self._ti(t, vectorized=False)), self.mass, method=method, **kwargs)[0]
        else:
            return np.zeros_like(self.pos(t=t))
        
    @_resolve_use_cached
    @_resolve_t
    def self_ax(self, t=..., use_cached=True, method=None, **kwargs):
        '''
        x-component of self-gravity acceleration on each particle in the component at *t*.
        
        Units: `kpc / Myr^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        use_cached : bool, optional
            Whether to use cached self-gravity if available. Default is True.
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON' (default): fast multipole method implemented in falcON.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method.

            For 'falcON', these include:
            - eps : float
                Gravitational softening length.
                Units: kpc
            - theta : float
                Tree opening angle for pyfalcon.
                Smaller = more accurate but slower.

            For 'direct', these include:
            - eps : float
                Gravitational softening length.
                Units: kpc
        
        Returns
        -------
        self_ax : (n_snaps, N) array
            x-component of self-gravity acceleration of 
            each particle at each snapshot.
            Units: kpc / Myr^2
        '''
        return self.self_gravity(t=t, method=method, use_cached=use_cached, **kwargs)[..., 0]
    
    @_resolve_use_cached
    @_resolve_t
    def self_ay(self, t=..., use_cached=True, method=None, **kwargs):
        '''
        y-component of self-gravity acceleration on each particle in the component at *t*.
        
        Units: `kpc / Myr^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        use_cached : bool, optional
            Whether to use cached self-gravity from integration
            if available. Default is True.
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON' (default): fast multipole method implemented in falcON.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method.

            For 'falcON', these include:
            - eps : float
                Gravitational softening length.
                Units: kpc
            - theta : float
                Tree opening angle for pyfalcon.
                Smaller = more accurate but slower.
        
        Returns
        -------
        self_ay : (n_snaps, N) array
            y-component of self-gravity acceleration of 
            each particle at each snapshot.
            Units: kpc / Myr^2
        '''
        return self.self_gravity(t=t, method=method, use_cached=use_cached, **kwargs)[..., 1]
    
    @_resolve_use_cached
    @_resolve_t
    def self_az(self, t=..., use_cached=True, method=None, **kwargs):
        '''
        z-component of self-gravity acceleration on each particle in the component at *t*.
        
        Units: `kpc / Myr^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON': Use the fast multipole method implemented in falcON.
            - 'direct': Use direct summation.
        use_cached : bool, optional
            Whether to use cached self-gravity from integration
            if available. Default is True.
        **kwargs
            Additional keyword arguments to pass to the gravity method.

            For 'falcON', these include:
            - eps : float
                Gravitational softening length.
                Units: kpc
            - theta : float
                Tree opening angle for pyfalcon.
                Smaller = more accurate but slower.
        
        Returns
        -------
        self_az : (n_snaps, N) array
            z-component of self-gravity acceleration of 
            each particle at each snapshot.
            Units: kpc / Myr^2
        '''
        return self.self_gravity(t=t, method=method, use_cached=use_cached, **kwargs)[..., 2]

    # --- External Acceleration --- #

    def external_acc(self, t=-1):
        '''
        Total external acceleration on each particle 
        in the component at *t*.
        Units: `kpc / Myr^2`

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        
        Returns
        -------
        ext_acc : (n_snaps, N, 3) array
            External acceleration of 
            each particle at each snapshot.
            Units: `kpc / Myr^2`
        '''
        ext_acc = np.zeros_like(self._velocities[self._ti(t)])
        for fn in self._ext_acc_fns:
            ext_acc += fn(self.pos(t=t), t=t)
        return ext_acc
    
    def external_ax(self, t=-1):
        '''
        x-component of external acceleration on each particle in the component at *t*.
        
        Units: `kpc / Myr^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        
        Returns
        -------
        external_ax : (n_snaps, N) array
            x-component of external acceleration of 
            each particle at each snapshot.
            Units: `kpc / Myr^2`
        '''
        return self.external_acc(t=t)[:, 0]

    def external_ay(self, t=-1):
        '''
        y-component of external acceleration on each particle in the component at *t*.
        
        Units: `kpc / Myr^2`

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        
        Returns
        -------
        external_ay : (n_snaps, N) array
            y-component of external acceleration of 
            each particle at each snapshot.
            Units: `kpc / Myr^2`
        '''
        return self.external_acc(t=t)[:, 1]
    
    def external_az(self, t=-1):
        '''
        z-component of external acceleration on each particle in the component at *t*.
        
        Units: `kpc / Myr^2`

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        
        Returns
        -------
        external_az : (n_snaps, N) array
            z-component of external acceleration of 
            each particle at each snapshot.
            Units: `kpc / Myr^2`
        '''        
        return self.external_acc(t=t)[:, 2]
    
    # --- Diagnostics -----------------------------------------------------------------
    @_resolve_use_cached
    def plot_diagnostic(self, method=None, use_cached=True, nsnap=None, 
                        filename=None, **kwargs):
        '''
        Plot global energy conservation as a function of 
        time.
        
        Parameters
        ----------
        method : str
            Method to use for computing self-gravity. Included options are:
            - 'falcON': fast multipole method implemented in falcON.
            - 'direct': direct summation.
        use_cached : bool, optional
            Whether to use cached self-gravity from integration if available. Default is True.
        nsnap : int, optional
            Number of snapshots to use. If None (default), will use all snapshots.
        filename : str, optional
            If provided, will save the plot to the given filename
            instead of showing it.
        **kwargs
            Additional keyword arguments to pass to the gravity method.
            For 'falcON', these include:
            - eps: Gravitational softening length (kpc)
            - theta: Tree opening angle
        '''
        import matplotlib.pyplot as plt

        skip_every = 1 if nsnap is None else max(1, len(self.times) // nsnap)
        
        plt.figure(figsize=(6,4))
        dE = self.dE(t=..., use_cached=use_cached, method=method, **kwargs)[::skip_every]
        plt.plot(self.times[::skip_every], dE, c='k')
        plt.yscale('log')
        plt.xlabel("Time (Myr)")
        plt.ylabel("$|\Delta E / E_0|$")
        plt.title("Energy Conservation")
        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    # --- Properties ---------------------------------------------------------------------
    @property
    def mass(self):
        return self._mass

    @property
    def times(self):
        return self._times

    # --- I/O ------------------------------------------------------------------------------

    def save(self):
        raise NotImplementedError("Saving and loading simulations is not yet supported.")
    
    def load(self):
        raise NotImplementedError("Saving and loading simulations is not yet supported.")

    # --- Output to external formats -------------------------------------------------------

    def to_galpy(self, t):
        raise NotImplementedError('Outputting galpt orbit is not yet supported.')