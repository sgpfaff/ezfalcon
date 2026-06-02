---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Computing Self-Gravity

```{code-cell} ipython3
:tags: [remove-cell]

# Setup (hidden): a small Plummer sphere to demonstrate on.
from tambora.simulation import Sim
from tambora.tools import mkPlummer_galpy

pos, vel, mass = mkPlummer_galpy(m=1e5, b=1.0, n=1000)

sim = Sim()
sim.add_particles("stars", pos=pos, vel=vel, mass=mass)
```

You can calculate the self-gravity acceleration within a
{class}`~tambora.simulation.Sim` instance with
{meth}`~tambora.simulation.Sim.self_gravity`. Before a simulation has been
run there are no cached forces, so pass a snapshot index ``t`` and a ``method``
to compute on the fly:

```{code-cell} ipython3
acc = sim.self_gravity(t=0, method="direct", eps=0.1)  # (N, 3) accelerations
acc.shape
```

Once a simulation has been run, the cached forces are used automatically and
the per-axis accessors become available:

```python
sim.self_gravity()  # (N, 3) array of accelerations
sim.self_ax()       # (N,) array of x-accelerations
sim.self_ay()       # (N,) array of y-accelerations
sim.self_az()       # (N,) array of z-accelerations
```

Or you can use the standalone {func}`~tambora.dynamics.self_gravity` function,
which takes arrays of positions, masses, and softening length(s) and returns
both the self-gravity acceleration **and** the potential:

```{code-cell} ipython3
from tambora.dynamics import self_gravity

acc, pot = self_gravity(pos, mass, eps=0.1)
acc.shape, pot.shape
```

## API

```{eval-rst}
.. autofunction:: tambora.dynamics.self_gravity

.. automethod:: tambora.simulation.Sim.self_gravity

.. automethod:: tambora.simulation.Sim.self_ax

.. automethod:: tambora.simulation.Sim.self_ay

.. automethod:: tambora.simulation.Sim.self_az

.. automethod:: tambora.simulation.Sim.self_potential
```
