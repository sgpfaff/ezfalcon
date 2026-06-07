User Guide
==========

In-depth explanations of tambora's core concepts. Start with **Simulations**
to build and run a model, dive into **Self-Gravity** for the force solvers, or
jump to **Tools & Interoperability** for everything around the edges.


All About Simulations
-----------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Setting Up a Simulation
      :link: making_a_simulation
      :link-type: doc

      Create a ``Sim``, add particle components, and access them.

   .. grid-item-card:: Generating Initial Conditions
      :link: initial_conditions
      :link-type: doc

      Sample Plummer, King, and NFW profiles — or any galpy DF.

   .. grid-item-card:: Adding External Forces and Potentials
      :link: external_forces
      :link-type: doc

      Apply custom, analytic, or time-dependent external forces and potentials.

   .. grid-item-card:: External Forces

   .. grid-item-card:: Running a Simulation
      :link: running_a_simulation
      :link-type: doc

      Configure the integrator and launch a run.

   .. grid-item-card:: Simulation Outputs
      :link: simulation_outputs
      :link-type: doc

      Access positions, velocities, energies, and diagnostics.


Forces
------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Force Types
      :link: force_types
      :link-type: doc

      External, self-gravity, and non-inertial forces.

   .. grid-item-card:: Self-Gravity Forces and Solvers
      :link: self_gravity_forces_and_solvers
      :link-type: doc

      Compute self-gravity forces with falcON, direct summation, or Barnes-Hut. Per-component self-gravity.
      Softening lengths. Supported Methods

   .. grid-item-card:: External Conservative Forces and Potentials
      :link: external_conservative_forces_and_potentials
      :link-type: doc

      Define and apply conservative forces from potentials.

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

