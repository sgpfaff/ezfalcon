import numpy as np
from ..dynamics.acceleration import self_gravity


class Component:
    """Slice view into one component's snapshot data."""

    def __init__(self, sim, sl, name):
        self._sim = sim
        self._name = name
        self._sl = sl

    def _snap(self, array, t):
        """Slice a (nsnaps, N, ...) array to this component at time t."""
        ti = self._sim._ti(t)
        if ti is ...:
            return array[:, self._sl]
        return array[ti, self._sl]

    def pos(self, t=...):
        '''
        Positions (x, y, z) of particles in the component at *t*.

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
        pos : (len(t), n_particles, 3) array or (n_particles,) array
            x, y, z positions at *t*.
            Units: `kpc`
        '''
        return self._snap(self._sim._positions, t)

     # --- Position Accessors -----------------------------------------------------------------

    def x(self, t=...):
        '''
        x-positions of all particles in the component at *t*.
        
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
        return self._snap(self._sim._positions, t)[..., 0]

    def y(self, t=...):
        '''
        y-positions of all particles in the component at *t*.
        
        Units: `kpc`
        '''
        return self._snap(self._sim._positions, t)[..., 1]

    def z(self, t=...):
        '''
        z-positions of all particles in the component at *t*.
        
        Units: `kpc`
        '''
        return self._snap(self._sim._positions, t)[..., 2]
    
     # --- Velocity Accessors -----------------------------------------------------------------

    def vel(self, t=...):
        '''
        Velocities (vx, vy, vz) of particles in the component at *t*.
        
        Units: `kpc/Myr`

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
            x, y, z velocities at *t*.
            Units: `kpc/Myr`
        '''
        return self._snap(self._sim._velocities, t)

    def vx(self, t=...):
        '''
        x-component of particle velocities in the component at *t*.
        
        Units: `kpc/Myr`

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
            x-velocities at *t*.
            Units: `kpc/Myr`
        '''
        return self._snap(self._sim._velocities, t)[..., 0]

    def vy(self, t=...):
        '''
        y-component of particle velocities in the component at *t*.
        
        Units: `kpc/Myr`

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
            y-velocities at *t*.
            Units: `kpc/Myr`
        '''
        return self._snap(self._sim._velocities, t)[..., 1]

    def vz(self, t=...):
        '''
        z-component of particle velocities in the component at *t*.
        
        Units: `kpc/Myr`

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
            z-velocities at *t*.
            Units: `kpc/Myr`
        '''
        return self._snap(self._sim._velocities, t)[..., 2]

    # --- Energy Accessors -----------------------------------------------------------------

    # --- Potential Energy --- #

    def compute_external_pot(self, t=...):
        '''
        External potential of particles in the 
        component at *t*.

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
        ext_pot = np.zeros(self.mass.shape[0])
        for fn in self._ext_pot_fns:
            ext_pot += fn(self.pos(t=t), t=t)
        return self.mass * ext_pot
    
    def compute_self_potential(self, t=-1, include_all_components=True, method='falcON', **kwargs):
        '''
        Self-gravitational potential energy of the 
        particles in the component at *t*. 
        
        Units:  `Msun kpc^2 / Myr^2`

        NOTE: This is the self-potential of the particles

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing the self-potential.
            If False, will only include the particles in this component when computing the self-potential.
            Default is True.
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
        if include_all_components:
            _, self_pot = self_gravity(self._sim.pos, self._sim.mass, method=method, **kwargs)[:, self._sl]
        else:
            _, self_pot = self_gravity(self.pos, self.mass, method=method, **kwargs)[:, self._sl]
        
        return self.mass * self_pot
    
    def PE(self, t=-1, include_all_components=True, method='falcON', **kwargs):
        '''
        Total potential energy of particles 
        in the component at *t*.

        Units:  `Msun kpc^2 / Myr^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing the self-potential energy.
            If False, will only include the particles in this component when computing the potential energy.
            Default is True.
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
        PE : (len(t), n_particles) array
            Total potential energy of each particle at each snapshot.
            Units: `Msun kpc^2 / Myr^2`
        
        '''
        if include_all_components:
            return self.PE(t=t, method=method, **kwargs)[:, self._sl]
        else:
            return self.compute_self_potential(t=t, include_all_components=False, method=method, **kwargs) + self.compute_external_pot(t=t)
    
    # --- Kinetic Energy --- #

    def KE(self, t=...):
        '''
        Kinetic energy of particles in the component at *t*.

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
        return 0.5 * self.mass * np.sum(self.vel(t=t) ** 2, axis=-1)

    # --- Total Energy --- #

    def energy(self, t=..., include_all_components=True, method='falcON', **kwargs):
        """
        Energy of the particles in the component at time t.
        
        Units:  `Msun kpc^2 / Myr^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
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
        return self.KE(t=t) + self.PE(t=t, include_all_components=include_all_components, method=method, **kwargs)
    
    def system_energy(self, t=-1, include_all_components=True, method='falcON', **kwargs):
        """
        Total energy of the component at time t.
        Units: Msun kpc^2/Myr^2

        E = Σ ½ mᵢ|vᵢ|² + ½ Σ mᵢΦ_self,ᵢ + Σ mᵢΦ_ext,ᵢ
        """
        return (self.KE(t=t).sum() + 
                0.5 * self.compute_self_potential(t=t, 
                                                  include_all_components=include_all_components, 
                                                  method=method, 
                                                  **kwargs).sum() 
                + np.sum(self.compute_external_pot(t=self._ti(t, vectorized=False)))
                )
    
    # --- Properties ---------------------------------------------------------------------- #
    
    @property
    def mass(self):
        return self._sim._mass[self._sl]


