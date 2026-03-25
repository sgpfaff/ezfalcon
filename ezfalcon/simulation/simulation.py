"""
Simulation class for ezfalcon.
"""

import time

import numpy as np
# from ..util import accept_vel, return_vel, G_INTERNAL
from .component import Component
from ..dynamics import integrate, self_gravity
import galpy
from ..util._galpy_bridge import _galpy_pot_to_acc_fn, _galpy_pot_to_pot_fn, _check_physical, _check_supported_pot


### ADD: Check _has_run decorator.
class Sim:
    """
    Self-gravitating N-body simulation powered by pyfalcon.
    """

    def __init__(self):
        self._host = None

        # Particle arrays -- built incrementally by add_particles()
        self._init_pos = None     # (N, 3) kpc
        self._init_vel = None     # (N, 3) kpc/Myr
        self._mass = None         # (N,) Msun
        self._slices = {}         # component name -> slice

        # Snapshot arrays -- populated by run()
        self._positions = None    # (n_snap, N, 3) kpc
        self._velocities = None   # (n_snap, N, 3) kpc/Myr

        #self._accelerations = None    # (n_snap, N, 3) kpc/Myr²
        self._self_acc = None # (n_snap, N, 3) kpc/Myr²
        #self._ext_acc = None # (n_snap, N, 3) kpc/Myr²

        #self._potentials = None    # (n_snap, N) kpc²/Myr²
        self._self_pot = None   # (n_snap, N)
        self._ext_pot = None   # (n_snap, N)
  
        self._times = None        # (n_snap,) Myr
        self._has_run = False
        self._self_gravity_on = True
        self._ext_acc_fns = {}     # list of functions that take (pos, t) and return (N, 3) acc
        self._ext_pot_fns = {}

    # ------------------------------------------------------------------ #
    #  Time index resolution                                              #
    # ------------------------------------------------------------------ #

    def _ti(self, t):
        """Resolve *t* to a snapshot index.

        None -> -1 (last), int -> direct index, float -> nearest time.
        """
        if isinstance(t, (int, np.integer)) or t == ...:
            return t
        else:
            return int(np.argmin(np.abs(self._times - t)))

    # ------------------------------------------------------------------ #
    #  Component access                                                   #
    # ------------------------------------------------------------------ #

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        slices = self.__dict__.get("_slices", {})
        if name in slices:
            return Component(self, slices[name])
        raise AttributeError(
            f"\'{type(self).__name__}\' has no attribute or component named {name!r}"
        )

    # ------------------------------------------------------------------ #
    #  Setup                                                              #
    # ------------------------------------------------------------------ #

    def add_particles(self, name, pos, vel, mass):
        """
        Add a named particle component.
        Provide (pos, vel, mass) directly [kpc, kpc/Myr, Msun].
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
        
        # if center is not None:
        #     center = np.asarray(center, dtype=np.float64)
        #     pos = pos + center[0]
        #     vel = vel + accept_vel(center[1])

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
        # self._potentials = np.zeros((1, N))
        self._times = np.array([0.0])

    # ------------------------------------------------------------------ #
    #  Run (stub)                                                         #
    # ------------------------------------------------------------------ #

    def run(self, t_end, dt, dt_out, eps, theta=0.6):
        """
        Run the simulation to *t_end* [Myr].
        """
        (self._positions, self._velocities,
         self._self_acc, self._self_pot, 
         self._times) = integrate(self._init_pos, self._init_vel, self._mass,
                                  self._self_gravity_on, self._ext_acc_fns,
                                  t_end, dt, dt_out, eps, theta=theta)
        self._has_run = True

    # ------------------------------------------------------------------ #
    #  Accessors -- all methods with optional t=                           #
    # ------------------------------------------------------------------ #

    def pos(self, t=...):
        '''
        Positions (x,y,z).
        Units: kpc

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        pos : (n_snaps, N, 3) array
            Positions at each snapshot.
            Units: kpc
        '''
        return self._positions[self._ti(t)]

    def vel(self, t=...):
        '''
        Velocities (vx,vy,vz).
        Units: kpc / Myr

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        vel : (n_snaps, N, 3) array
            Velocities at each snapshot.
            Units: kpc / Myr
        '''
        return self._velocities[self._ti(t)]

    def x(self, t=...):
        '''
        x-positions of all particles at time t.
        Units: kpc

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        x : (n_snaps, N) array
            x-positions at each snapshot.
            Units: kpc
        '''
        return self._positions[self._ti(t), :, 0]
 
    def y(self, t=...):
        '''
        y-positions of all particles at time t.
        Units: kpc

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        y : (n_snaps, N) array
            y-positions at each snapshot.
            Units: kpc
        '''
        return self._positions[self._ti(t), :, 1]

    def z(self, t=...):
        '''
        z-positions of all particles at time t.
        Units: kpc

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        z : (n_snaps, N) array
            z-positions at each snapshot.
            Units: kpc
        '''
        return self._positions[self._ti(t), :, 2]

    def vx(self, t=...):
        '''
        x-component of particle velocities at time t.
        Units: kpc / Myr

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        vx : (n_snaps, N) array
            x-component of velocities at each snapshot.
            Units: kpc / Myr
        '''
        return self._velocities[self._ti(t), :, 0]

    def vy(self, t=...):
        '''
        y-component of particle velocities at time t.
        Units: kpc / Myr

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        vy : (n_snaps, N) array
            y-component of velocities at each snapshot.
            Units: kpc / Myr
        '''
        return self._velocities[self._ti(t), :, 1]

    def vz(self, t=...):
        '''
        z-component of particle velocities at time t.
        Units: kpc / Myr

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        vz : (n_snaps, N) array
            z-component of velocities at each snapshot.
            Units: kpc / Myr
        '''
        return self._velocities[self._ti(t), :, 2]
    
    ### Energy Accessors ###

    def compute_external_pot(self, t=-1):
        '''
        External potential at time t.
        Currently only supports time-independent external potentials.
        Units: kpc^2 / Myr^2

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        ext_pot : (n_snaps, N) array
            External potential at each snapshot.
            Units: kpc^2 / Myr^2
        '''
        ext_pot = np.zeros(self._mass.shape[0])
        for fn in self._ext_pot_fns.values():
            ext_pot += fn(self.pos(t=t), t=t)
        return ext_pot

    def self_PE(self, t=...):
        '''
        Self-gravitational potential energy of each particle at time t.
        Units:  Msun kpc^2 / Myr^2

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        self_PE : (n_snaps, N) array
            Self-gravitational potential energy of each particle at each snapshot.
            Units: Msun kpc^2 / Myr^2
        '''
        return self._mass * self._self_pot[self._ti(t)]
    
    def PE(self, t=...):
        '''
        Total potential energy of each particle at time t.
        Units:  Msun kpc^2 / Myr^2

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            
            Default is ... (ellipsis), which returns the value at all times.
        
        Returns
        -------
        PE : (n_snaps, N) array
            Total potential energy of each particle at each snapshot.
            Units: Msun kpc^2 / Myr^2
        '''
        return self.self_PE(t=t) + self._mass * self.compute_external_pot(t=t)
    
    def KE(self, t=...):
        '''
        Kinetic energy of each particle at time t.
        Units:  Msun kpc^2 / Myr^2

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        KE : (n_snaps, N) array
            Kinetic energy of each particle at each snapshot.
            Units: Msun kpc^2 / Myr^2
        '''
        return 0.5 * self._mass * np.sum(self.vel(t=t) ** 2, axis=-1)

    def energy(self, t=...):
        """
        Energy of each particle at time t.
        Units:  Msun kpc^2 / Myr^2

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            
            Default is ... (ellipsis), which returns the value at all times.
        
        Returns
        -------
        energy : (n_snaps, N) array
            Total energy of each particle at each snapshot.
            Units: Msun kpc^2 / Myr^2
        """
        return self.KE(t=t) + self.PE(t=t)
    
    def system_energy(self, t=-1):
        """
        Total conserved system energy at time t.
        Units: Msun kpc^2 / Myr^2

        E = Σ ½ mᵢ|vᵢ|² + ½ Σ mᵢΦ_self,ᵢ + Σ mᵢΦ_ext,ᵢ

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.

            Default is ... (ellipsis), which returns the value at all times.
        
        Returns
        -------
        energy : float
            Total energy of the system at time t.
            Units: Msun kpc^2 / Myr^2
        """
        return np.sum(self.KE(t=t)) + 0.5 * np.sum(self.self_PE(t=t)) + np.sum(self._mass * self.compute_external_pot(t=t))
    
    def dE(self):
        '''
        Percent change in total energy over the simulation time.

        Returns
        -------
        dE : (n_snaps,) array
            Percent change in total energy at each snapshot.
        '''
        Es = np.array([self.system_energy(t=i) for i in range(len(self.times))])
        return np.abs((Es - Es[0]) / Es[0])
    
    def add_external_pot(self, name, pot):
        '''
        Add an external potential to the simulation.

        Parameters
        ----------
        name : str
            Name of the external potential.
        pot : galpy.potential.Potential
            External potential to add.
        '''
        if isinstance(pot, galpy.potential.Potential):
            _check_supported_pot(pot)
            _check_physical(pot)
            self._ext_pot_fns[name] = _galpy_pot_to_pot_fn(pot)
            self._ext_acc_fns[name] = _galpy_pot_to_acc_fn(pot)
        else:
            raise TypeError("External potential must be a galpy Potential object.")
    
    #### Self-Gravity ###

    def turn_self_gravity_on(self):
        self._self_gravity_on = True

    def turn_self_gravity_off(self):
        self._self_gravity_on = False


    def compute_self_gravity(self, eps, theta, t=...):
        '''
        Compute the self-gravity acceleration of each 
        particle in the component at time t.
        Units: kpc / Myr^2

        Parameters
        ----------
        eps : float
            Gravitational softening length.
            Units: kpc
        theta : float
            Tree opening angle for pyfalcon.
            Smaller = more accurate but slower.
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        
        Returns
        -------
        self_acc : (n_snaps, N, 3) array
            Self-gravity acceleration of each particle at each snapshot.
            Units: kpc / Myr^2
        '''
        if self._self_gravity_on:
            return self_gravity(self.pos(t=t), self.mass, eps=eps, theta=theta)[0]
        else:
            return np.zeros_like(self.pos(t=t))
    

    ### Acceleration Accessors ###

    def self_gravity_acc(self, t=...):
        '''
        Self-gravity acceleration of each particle in 
        the component at time t, as calculated during 
        integration.
        Units: kpc / Myr^2

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        
        Returns
        -------
        self_acc : (n_snaps, N, 3) array
            Self-gravity acceleration of each particle at each snapshot.
            Units: kpc / Myr^2
        '''
        return self._self_acc[self._ti(t)]
    
    def self_ax(self, t=...):
        '''
        x-component of self-gravity acceleration on each particle in the component at time t.
        Units: kpc / Myr^2

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        
        Returns
        -------
        self_ax : (n_snaps, N) array
            x-component of self-gravity acceleration of 
            each particle at each snapshot.
            Units: kpc / Myr^2
        '''
        return self._self_acc[self._ti(t), :, 0]
    
    def self_ay(self, t=...):
        '''
        y-component of self-gravity acceleration on each particle in the component at time t.
        Units: kpc / Myr^2

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        
        Returns
        -------
        self_ay : (n_snaps, N) array
            y-component of self-gravity acceleration of 
            each particle at each snapshot.
            Units: kpc / Myr^2
        '''
        return self._self_acc[self._ti(t), :, 1]
    
    def self_az(self, t=...):
        '''
        z-component of self-gravity acceleration on each particle in the component at time t.
        Units: kpc / Myr^2

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        
        Returns
        -------
        self_az : (n_snaps, N) array
            z-component of self-gravity acceleration of 
            each particle at each snapshot.
            Units: kpc / Myr^2
        '''
        return self._self_acc[self._ti(t), :, 2]

    def external_acc(self, t=-1):
        '''
        Total external acceleration on each particle 
        in the component at time t.
        Units: kpc / Myr^2

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        
        Returns
        -------
        ext_acc : (n_snaps, N, 3) array
            External acceleration of 
            each particle at each snapshot.
            Units: kpc / Myr^2
        '''
        ext_acc = np.zeros_like(self._velocities[self._ti(t)])
        for fn in self._ext_acc_fns.values():
            ext_acc += fn(self.pos(t=t), t=t)
        return ext_acc
    
    def external_ax(self, t=-1):
        '''
        x-component of external acceleration on each particle in the component at time t.
        Units: kpc / Myr^2

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        
        Returns
        -------
        external_ax : (n_snaps, N) array
            x-component of external acceleration of 
            each particle at each snapshot.
            Units: kpc / Myr^2
        '''
        return self.external_acc(t=t)[:, 0]

    def external_ay(self, t=-1):
        '''
        y-component of external acceleration on each particle in the component at time t.
        Units: kpc / Myr^2

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        
        Returns
        -------
        external_ay : (n_snaps, N) array
            y-component of external acceleration of 
            each particle at each snapshot.
            Units: kpc / Myr^2
        '''
        return self.external_acc(t=t)[:, 1]
    
    def external_az(self, t=-1):
        '''
        z-component of external acceleration on each particle in the component at time t.
        Units: kpc / Myr^2

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        
        Returns
        -------
        external_az : (n_snaps, N) array
            z-component of external acceleration of 
            each particle at each snapshot.
            Units: kpc / Myr^2
        '''        
        return self.external_acc(t=t)[:, 2]
    
    def plot_diagnostic(self, filename=None):
        '''
        Plot global energy conservation as a function of 
        time.
        
        Parameters
        ----------
        filename : str, optional
            If provided, will save the plot to the given filename
            instead of showing it.
        '''
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,4))
        dE = self.dE()
        plt.plot(self.times, dE, c='k')
        plt.yscale('log')
        plt.xlabel("Time (Myr)")
        plt.ylabel("$|\Delta E / E_0|$")
        plt.title("Energy Conservation")
        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def relax(self, components, t_end, dt, eps, theta=0.6, force_center=True):
        '''
        Relax the given components in isolation for time t_end.

        Parameters
        ----------
        components : list of str or 'all'
            Names of components to relax. If 'all', relax all components.
        t_end : float
            End time of the relaxation.
            Units: Myr
        dt : float 
            Timestep for the relaxation.
            Units: Myr
        eps : float
            Softening length for integration.
            Units: kpc
        theta : float, optional
            Opening angle for the integration.
        force_center : bool, optional
            If True, will return the center of the relaxed component
            to its initial position and velocity to prevent drifting.
        '''
        if components == 'all':
            if force_center:
                orig_center_pos = np.mean(self._init_pos, axis=0)
                orig_center_vel = np.mean(self._init_vel, axis=0)
            all_pos, all_vel, _, _, _ = integrate(self._init_pos, self._init_vel, self._mass, 
                      True, {}, t_end, dt, dt_out=t_end, eps=eps, theta=theta)
            self._init_pos = all_pos[-1]
            self._init_vel = all_vel[-1]
            if force_center:
                self._init_pos += orig_center_pos - np.mean(self._init_pos, axis=0)
                self._init_vel += orig_center_vel - np.mean(self._init_vel, axis=0)
        else:
            comp_pos, comp_vel, comp_masses = [], [], []
            new_slices = {}
            offset = 0
            for comp in components:
                if comp not in self._slices:
                    raise ValueError(f'Component \'{comp}\' not included in the simulation.')
                else:
                    sl = self._slices[comp]
                    comp_pos.append(self._init_pos[sl])
                    comp_vel.append(self._init_vel[sl])
                    comp_masses.append(self._mass[sl])
                    n = len(self._mass[sl])
                    new_slices[comp] = slice(offset, offset + n)
                    offset += n
            comp_pos = np.concatenate(comp_pos)
            comp_vel = np.concatenate(comp_vel)
            comp_masses = np.concatenate(comp_masses)

            new_positions, new_velocities, _, _, _ = integrate(comp_pos, comp_vel, comp_masses, True,
                                                               {}, t_end, dt, dt_out=t_end, eps=eps, theta=theta)
            for comp in components:
                sl = self._slices[comp]
                if force_center:
                    center_pos = np.mean(self._init_pos[sl], axis=0)
                    center_vel = np.mean(self._init_vel[sl], axis=0)
                    new_positions[-1, new_slices[comp]] += center_pos - np.mean(new_positions[-1, new_slices[comp]], axis=0)
                    new_velocities[-1, new_slices[comp]] += center_vel - np.mean(new_velocities[-1, new_slices[comp]], axis=0)
                self._init_pos[sl] = new_positions[-1, new_slices[comp]]
                self._init_vel[sl] = new_velocities[-1, new_slices[comp]]
    @property
    def mass(self):
        return self._mass

    @property
    def times(self):
        return self._times

    # ------------------------------------------------------------------ #
    #  I/O                                                                #
    # ------------------------------------------------------------------ #

    # def save(self, path):
    #     import h5py
    #     with h5py.File(path, "w") as f:
    #         f.create_dataset("times", data=self._times)
    #         f.create_dataset("positions", data=self._positions)
    #         f.create_dataset("velocities", data=self._velocities)
    #         f.create_dataset("potentials", data=self._potentials)
    #         f.create_dataset("self_potentials", data=self._self_potentials)
    #         f.create_dataset("mass", data=self._mass)
    #         grp = f.create_group("components")
    #         for cname, sl in self._slices.items():
    #             grp.attrs[cname] = [sl.start, sl.stop]

    # @classmethod
    # def load(cls, path):
    #     import h5py
    #     with h5py.File(path, "r") as f:
    #         sim = cls()
    #         sim._times = f["times"][:]
    #         sim._positions = f["positions"][:]
    #         sim._velocities = f["velocities"][:]
    #         sim._potentials = f["potentials"][:]
    #         sim._self_potentials = f["self_potentials"][:]
    #         sim._mass = f["mass"][:]
    #         sim._slices = {
    #             name: slice(int(v[0]), int(v[1]))
    #             for name, v in f["components"].attrs.items()
    #         }
    #         sim._has_run = True
    #     return sim

    # ------------------------------------------------------------------ #
    #  Converting to galpy and Agama                                     #
    # ------------------------------------------------------------------ #

    # def to_galpy(self, t):
    #     raise NotImplementedError('Outputting galpt orbit is not yet supported.')

    # def to_agama(self, t):
    #     raise NotImplementedError('Outputting agama snapshot is not yet supported.')