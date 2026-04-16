Simulation Outputs
==================

Accessing Snapshots
-------------------

All accessor methods accept a ``t`` parameter that controls which snapshot(s)
are returned:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Behaviour
   * - ``int``
     - Snapshot by index (supports negative indexing, e.g. ``t=-1`` for last)
   * - ``float``
     - Closest snapshot to the given time in Gyr
   * - ``...`` (Ellipsis)
     - All snapshots, returned as ``(nsnap, N, ...)``


Position Accessors
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 55 15

   * - Method
     - Description
     - Units
   * - :meth:`~ezfalcon.simulation.Sim.pos`
     - Full position vector (N, 3)
     - kpc
   * - :meth:`~ezfalcon.simulation.Sim.x`
     - Cartesian x
     - kpc
   * - :meth:`~ezfalcon.simulation.Sim.y`
     - Cartesian y
     - kpc
   * - :meth:`~ezfalcon.simulation.Sim.z`
     - Cartesian z
     - kpc
   * - :meth:`~ezfalcon.simulation.Sim.r`
     - Spherical radius
     - kpc
   * - :meth:`~ezfalcon.simulation.Sim.phi`
     - Azimuthal angle
     - rad
   * - :meth:`~ezfalcon.simulation.Sim.theta`
     - Polar angle
     - rad
   * - :meth:`~ezfalcon.simulation.Sim.cylR`
     - Cylindrical radius
     - kpc


Velocity Accessors
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 55 15

   * - Method
     - Description
     - Units
   * - :meth:`~ezfalcon.simulation.Sim.vel`
     - Full velocity vector (N, 3)
     - km/s
   * - :meth:`~ezfalcon.simulation.Sim.vx`
     - Cartesian vx
     - km/s
   * - :meth:`~ezfalcon.simulation.Sim.vy`
     - Cartesian vy
     - km/s
   * - :meth:`~ezfalcon.simulation.Sim.vz`
     - Cartesian vz
     - km/s
   * - :meth:`~ezfalcon.simulation.Sim.vr`
     - Spherical radial velocity
     - km/s
   * - :meth:`~ezfalcon.simulation.Sim.vphi`
     - Azimuthal angular velocity
     - km/s/kpc
   * - :meth:`~ezfalcon.simulation.Sim.vtheta`
     - Polar velocity
     - km/s
   * - :meth:`~ezfalcon.simulation.Sim.cylvR`
     - Cylindrical radial velocity
     - km/s


Momentum Accessors
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 55 15

   * - Method
     - Description
     - Units
   * - :meth:`~ezfalcon.simulation.Sim.p`
     - Full momentum vector (N, 3)
     - Msun km/s
   * - :meth:`~ezfalcon.simulation.Sim.px`
     - Momentum x-component
     - Msun km/s
   * - :meth:`~ezfalcon.simulation.Sim.py`
     - Momentum y-component
     - Msun km/s
   * - :meth:`~ezfalcon.simulation.Sim.pz`
     - Momentum z-component
     - Msun km/s


Angular Momentum Accessors
--------------------------

All angular momentum methods also accept ``center_pos`` and ``center_vel``
keyword arguments.

.. list-table::
   :header-rows: 1
   :widths: 15 55 15

   * - Method
     - Description
     - Units
   * - :meth:`~ezfalcon.simulation.Sim.L`
     - Full angular momentum vector (N, 3)
     - Msun kpc km/s
   * - :meth:`~ezfalcon.simulation.Sim.Lx`
     - Angular momentum x-component
     - Msun kpc km/s
   * - :meth:`~ezfalcon.simulation.Sim.Ly`
     - Angular momentum y-component
     - Msun kpc km/s
   * - :meth:`~ezfalcon.simulation.Sim.Lz`
     - Angular momentum z-component
     - Msun kpc km/s


Energy Accessors
----------------

.. list-table::
   :header-rows: 1
   :widths: 15 55 15

   * - Method
     - Description
     - Units
   * - :meth:`~ezfalcon.simulation.Sim.KE`
     - Kinetic energy per particle
     - Msun km²/s²
   * - :meth:`~ezfalcon.simulation.Sim.self_potential`
     - Self-gravitational potential energy per particle
     - Msun km²/s²
   * - :meth:`~ezfalcon.simulation.Sim.compute_external_pot`
     - External potential energy per particle
     - Msun km²/s²
   * - :meth:`~ezfalcon.simulation.Sim.PE`
     - Total potential energy per particle (self + external)
     - Msun km²/s²
   * - :meth:`~ezfalcon.simulation.Sim.energy`
     - Total energy per particle (KE + PE)
     - Msun km²/s²
   * - :meth:`~ezfalcon.simulation.Sim.system_energy`
     - Total system energy (sum over all particles)
     - Msun km²/s²
   * - :meth:`~ezfalcon.simulation.Sim.dE`
     - Fractional energy change \|ΔE/E₀\|
     - dimensionless


Acceleration Accessors
----------------------

.. list-table::
   :header-rows: 1
   :widths: 15 55 15

   * - Method
     - Description
     - Units
   * - :meth:`~ezfalcon.simulation.Sim.self_gravity`
     - Self-gravity acceleration vector (N, 3)
     - km/s²
   * - :meth:`~ezfalcon.simulation.Sim.self_ax`
     - Self-gravity x-component
     - km/s²
   * - :meth:`~ezfalcon.simulation.Sim.self_ay`
     - Self-gravity y-component
     - km/s²
   * - :meth:`~ezfalcon.simulation.Sim.self_az`
     - Self-gravity z-component
     - km/s²
   * - :meth:`~ezfalcon.simulation.Sim.external_acc`
     - External acceleration vector (N, 3)
     - km/s²
   * - :meth:`~ezfalcon.simulation.Sim.external_ax`
     - External acceleration x-component
     - km/s²
   * - :meth:`~ezfalcon.simulation.Sim.external_ay`
     - External acceleration y-component
     - km/s²
   * - :meth:`~ezfalcon.simulation.Sim.external_az`
     - External acceleration z-component
     - km/s²


Making Diagnostic Plots
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Description
   * - :meth:`~ezfalcon.simulation.Sim.plot_energy_diagnostic`
     - Plot fractional energy change as a function of time
   * - :meth:`~ezfalcon.simulation.Sim.plot_momentum_diagnostic`
     - Plot momentum conservation

.. _component_accessors:

Component Accessors
-------------------

Each named component is accessible as an attribute of the ``Sim`` object
(e.g. ``sim.stars``). The returned :class:`~ezfalcon.simulation.Component`
object provides the same accessor interface as ``Sim``, scoped to that
component's particles.

🚧 *Still working on it...*


Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 15 55 15

   * - Property
     - Description
     - Units
   * - :attr:`~ezfalcon.simulation.Sim.mass`
     - Particle masses
     - Msun
   * - :attr:`~ezfalcon.simulation.Sim.times`
     - Snapshot times
     - Gyr


API
---

.. autoclass:: ezfalcon.simulation.Sim
   :members:
   :undoc-members:
   :exclude-members: add_particles, run, add_external_pot, add_external_acc

.. autoclass:: ezfalcon.simulation.Component
   :members:
   :undoc-members:
