import numpy as np
#from .units import return_vel

class Component:
    """Slice view into one component's snapshot data."""

    def __init__(self, sim, sl):
        self._sim = sim
        self._sl = sl

    def pos(self, t=...):
        '''
        Positions (x, y, z) of particles in the component.
        Units: kpc
        '''
        return self._sim._positions[self._sim._ti(t), self._sl]

    def vel(self, t=...):
        '''
        Velocities (vx, vy, vz) of particles in the component.
        '''
        return self._sim._velocities[self._sim._ti(t), self._sl]

    def x(self, t=...):
        '''
        x-positions of all particles in the component at time t.
        Units: kpc
        '''
        return self._sim._positions[self._sim._ti(t), self._sl, 0]

    def y(self, t=...):
        '''
        y-positions of all particles in the component at time t.
        Units: kpc
        '''
        return self._sim._positions[self._sim._ti(t), self._sl, 1]

    def z(self, t=...):
        '''
        z-positions of all particles in the component at time t.
        Units: kpc
        '''
        return self._sim._positions[self._sim._ti(t), self._sl, 2]

    def vx(self, t=...):
        '''
        x-component of particle velocities in the 
        component at time t.
        Units: kpc/Myr
        '''
        return self._sim._velocities[self._sim._ti(t), self._sl, 0]

    def vy(self, t=...):
        '''
        y-component of particle velocities in the 
        component at time t.
        Units: kpc/Myr
        '''
        return self._sim._velocities[self._sim._ti(t), self._sl, 1]

    def vz(self, t=...):
        '''
        z-component of particle velocities in the 
        component at time t.
        Units: kpc/Myr
        '''
        return self._sim._velocities[self._sim._ti(t), self._sl, 2]

    def external_pot(self, t=...):
        pass
    
    def self_PE(self, t=...):
        '''
        Self-gravitational potential energy of each particle 
        in the component at time t.
        Units:  Msun kpc²/Myr²
        '''
        return self._sim._mass[self._sl] * self._sim._self_pot[self._sim._ti(t), self._sl]
    
    def PE(self, t=...):
        '''
        Total potential energy of each particle 
        in the component at time t.
        Units:  Msun kpc²/Myr²
        '''
        return self.self_PE(t=t) 
    
    def KE(self, t=...):
        '''
        Kinetic energy of each particle 
        in the component at time t.
        Units:  Msun kpc²/Myr²
        '''
        return 0.5 * self.mass * np.sum(self.vel(t=t) ** 2, axis=-1)

    def energy(self, t=...):
        """
        Energy of each particle in the 
        component at time t.
        Units:  Msun kpc²/Myr²
        """
        return self.KE(t=t) + self.PE(t=t)
    
    def system_energy(self, t=...):
        """
        Total energy of the component at time t.
        Units: Msun kpc²/Myr²

        E = Σ ½ mᵢ|vᵢ|² + ½ Σ mᵢΦ_self,ᵢ + Σ mᵢΦ_ext,ᵢ
        """
        return self.KE(t=t).sum() + 0.5 * self.self_PE(t=t).sum()
    
    # def dE(self):
    #     '''
    #     Percent change in total energy over the simulation time.
    #     Units:  Msun kpc²/Myr²
    #     '''
    #     Es = np.array([self.system_energy(t=i) for i in range(len(self._sim.times))])
    #     return np.abs((Es - Es[0]) / Es[0])

    @property
    def mass(self):
        return self._sim._mass[self._sl]
