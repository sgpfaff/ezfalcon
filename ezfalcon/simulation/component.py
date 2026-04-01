import numpy as np
from ..dynamics.acceleration import self_gravity

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
        print(self._sim._positions.shape)
        print(self._sim._positions[:, self._sl].shape)
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
        ext_pot = np.zeros(self._mass.shape[0])
        for fn in self._ext_pot_fns:
            ext_pot += fn(self.pos(t=t), t=t)
        return self._mass * ext_pot
    
    def compute_self_potential(self, eps, theta, t=-1):
        '''
        Self-gravitational potential energy of each particle 
        in the component at time t.
        Units:  Msun kpc²/Myr²
        '''
        _, self_pot = self_gravity(self.pos(t=self._ti(t, vectorized=True)), self._mass, eps, theta=theta)
        return self._sim._mass[self._sl] * self_pot
    
    def PE(self, eps, theta, t=-1):
        '''
        Total potential energy of each particle 
        in the component at time t.
        Units:  Msun kpc²/Myr²
        '''
        return self.PE(t=t, eps=eps, theta=theta)[self._sl]
    
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
        return self.KE(t=t) + self.PE(t=t, eps=0, theta=0)
    
    def system_energy(self, t=-1, eps=0, theta=0):
        """
        Total energy of the component at time t.
        Units: Msun kpc^2/Myr^2

        E = Σ ½ mᵢ|vᵢ|² + ½ Σ mᵢΦ_self,ᵢ + Σ mᵢΦ_ext,ᵢ
        """
        return self.KE(t=t).sum() + 0.5 * self.compute_self_potential(t=t, eps=eps, theta=theta).sum() + np.sum(self.compute_external_pot(t=self._ti(t, vectorized=False)))

    @property
    def mass(self):
        return self._sim._mass[self._sl]
