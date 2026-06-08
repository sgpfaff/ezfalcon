User Guide
==========

In-depth explanations of tambora's core concepts. Start with **All About Simulations**
to build and run a simulation, dive into **Forces** for an understanding of how self-gravity and external forces are handled, or
jump to **Tools & Interoperability** for everything around the edges.


All About Simulations
--------------------

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Setting Up a Simulation
      :link: setting_up_a_simulation
      :link-type: doc

      Instantiating a simulation, adding particles and external forces, toggling self-gravity.

   .. grid-item-card:: Running a Simulation
      :link: running_a_simulation
      :link-type: doc

      Configure the integrator and launch a run.

   .. grid-item-card:: Simulation Accessors 
      :link: simulation_outputs
      :link-type: doc

      Access positions, velocities, energies, and diagnostics.
   
   .. grid-item-card:: Generating Initial Conditions
      :link: generating_initial_conditions
      :link-type: doc
      :img-bottom: sampled_plummer_IC.png

      Generate equilibrium initial conditions for with tambora's built-in IC generation convenience functions.


   .. grid-item-card:: Including Multiple Sets of Particles
      :link: including_multiple_sets_of_particles
      :link-type: doc
      :img-bottom: multicomponent_IC.png

      Adding, simulating, and accessing multiple sets of particles in a simulation.


Forces
------

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Types of Forces
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

.. grid:: 1 2 2 3
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
   :hidden:
   :caption: All About Simulations
   :maxdepth: 1

   setting_up_a_simulation
   running_a_simulation
   simulation_outputs
   including_multiple_sets_of_particles
   generating_initial_conditions

.. toctree::
   :hidden:
   :caption: Forces
   :maxdepth: 1

   force_types
   self_gravity_forces_and_solvers
   external_conservative_forces_and_potentials

.. toctree::
   :hidden:
   :caption: Tools & Interoperability
   :maxdepth: 1

   satellite_tools
   interoperability
   units
