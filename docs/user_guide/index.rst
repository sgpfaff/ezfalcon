User Guide
==========

In-depth explanations of tambora's core concepts. Start with **Simulations**
to build and run a model, dive into **Self-Gravity** for the force solvers, or
jump to **Tools & Interoperability** for everything around the edges.


Simulations
-----------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Making a Simulation
      :link: making_a_simulation
      :link-type: doc

      Create a ``Sim``, add particle components, and access them.

   .. grid-item-card:: Initial Conditions
      :link: initial_conditions
      :link-type: doc

      Sample Plummer, King, and NFW profiles — or any galpy DF.

   .. grid-item-card:: External Potentials
      :link: external_potentials
      :link-type: doc

      Drive orbits with galpy potentials, from spherical to triaxial.

   .. grid-item-card:: External Forces
      :link: external_forces
      :link-type: doc

      Apply custom, analytic, or time-dependent external forces.

   .. grid-item-card:: Running a Simulation
      :link: running_a_simulation
      :link-type: doc

      Configure the integrator and launch a run.

   .. grid-item-card:: Simulation Outputs
      :link: simulation_outputs
      :link-type: doc

      Access positions, velocities, energies, and diagnostics.


Self-Gravity
------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Computing Self-Gravity
      :link: computing_self_gravity
      :link-type: doc

      Compute self-gravity accelerations and the potential.

   .. grid-item-card:: Per-Component Self-Gravity
      :link: computing_component_self_gravity
      :link-type: doc

      Restrict self-gravity to individual components.

   .. grid-item-card:: Softening Lengths
      :link: softening_lengths
      :link-type: doc

      Choose scalar or per-particle gravitational softening.

   .. grid-item-card:: Supported Methods
      :link: self_gravity_methods
      :link-type: doc

      falcON, direct summation, and Barnes-Hut tree solvers.


Tools & Interoperability
------------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Satellite Tools
      :link: satellite_tools
      :link-type: doc

      Track satellites, bound mass, and tidal debris.

   .. grid-item-card:: Interoperability
      :link: interoperability
      :link-type: doc

      Move data to and from galpy, astropy, and friends.

   .. grid-item-card:: Units
      :link: units
      :link-type: doc

      tambora's internal and user-facing unit conventions.


.. The toctrees below populate the left sidebar, grouped by caption. They
.. replace the old stub pages (simulation.rst / self_gravity.rst) that existed
.. only to hold a nested toctree.

.. toctree::
   :caption: Simulations
   :maxdepth: 1
   :hidden:

   making_a_simulation
   initial_conditions
   external_potentials
   external_forces
   running_a_simulation
   simulation_outputs

.. toctree::
   :caption: Self-Gravity
   :maxdepth: 1
   :hidden:

   computing_self_gravity
   computing_component_self_gravity
   softening_lengths
   self_gravity_methods

.. toctree::
   :caption: Tools & Interoperability
   :maxdepth: 1
   :hidden:

   satellite_tools
   interoperability
   units
