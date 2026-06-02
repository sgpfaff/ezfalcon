---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Making a Simulation

The {class}`~tambora.simulation.Sim` class is the core of {ref}`home`. It
orchestrates building, running, and analyzing n-body simulations and is what
you will interact with the most as a result. Instantiating a simulation is
super simple:

```{code-cell} ipython3
from tambora.simulation import Sim

sim = Sim()
```

## Adding Particles

To add particles to the simulation, use the
{meth}`~tambora.simulation.Sim.add_particles` method. This method takes a
component name (e.g. ``'stars'``, ``'dark_matter'``) and arrays of positions,
velocities, and masses. Here we generate a quick Plummer sphere for the
``'stars'`` component (see [](initial_conditions.md) for the sampling tools):

```{code-cell} ipython3
from tambora.tools import mkPlummer_galpy

pos, vel, mass = mkPlummer_galpy(m=1e5, b=1.0, n=1000)
sim.add_particles("stars", pos=pos, vel=vel, mass=mass)
```

## Adding Multiple Components

You can add as many components as you'd like following the same procedure, as
long as you give each a unique name. For example, we add a ``'dark_matter'``
component alongside the stars:

```{code-cell} ipython3
dm_pos, dm_vel, dm_mass = mkPlummer_galpy(m=1e7, b=3.0, n=2000)
sim.add_particles("dark_matter", pos=dm_pos, vel=dm_vel, mass=dm_mass)
```

You can access the properties of individual components using the component
name. Each returns an array shaped ``(nsnap, N, ...)``:

```{code-cell} ipython3
star_positions = sim.stars.pos()
dm_velocities = sim.dark_matter.vel()

star_positions.shape, dm_velocities.shape
```

You can also still access all particles at once, for example:

```{code-cell} ipython3
all_positions = sim.pos()
all_positions.shape
```

Methods for accessing the properties of individual components are discussed in
more detail in the {ref}`component_accessors` section of the user guide.

## API

```{eval-rst}
.. autoclass:: tambora.simulation.Sim

.. automethod:: tambora.simulation.Sim.add_particles
```
