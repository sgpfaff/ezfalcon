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

      Instantiating a simulation.

   .. grid-item-card:: Generating Initial Conditions
      :link: generating_initial_conditions
      :link-type: doc
      :img-bottom: sampled_plummer_IC.png

      Generate equilibrium initial conditions for with tambora's built-in IC generation convenience functions.

   .. grid-item-card:: Adding Particles to a Simulation
      :link: adding_particles_to_a_sim
      :link-type: doc
      :img-bottom: multicomponent_IC.png

      Adding a set (or sets) of particles to a simulation.

   .. grid-item-card:: Adding External Forces to a Simulation
      :link: adding_external_forces_to_a_sim
      :link-type: doc

      Adding external forces to a simulation, including conservative forces from potentials and non-conservative forces from user-defined force classes.

   .. grid-item-card:: Running a Simulation
      :link: running_a_simulation
      :link-type: doc

      Configure the integrator and launch a run.

   .. grid-item-card:: Integrators
      :link: integrators
      :link-type: doc

      Overview of how integrators are implemented in tambora and the options available to users for customization.

   .. grid-item-card:: Simulation Accessors 
      :link: simulation_outputs
      :link-type: doc

      Access particle positions, velocities, energies, accelerations, and more from a simulation.
   
   .. grid-item-card:: Evaluating Properties of Multiple Sets of Particles
      :link: evaluating_properties_of_multiple_sets_of_particles
      :link-type: doc
      

      Accessing the properties of multiple sets of particles in a simulation.



Forces
------

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Overview of Force Classes
      :link: force_types
      :link-type: doc

      External, self-gravity, and non-inertial forces.

   .. grid-item-card:: Self-Gravity Force Classes
      :link: self_gravity_force_and_solvers
      :link-type: doc
      :img-bottom: tree_visualization.gif

      Compute self-gravity forces with falcON, direct summation, or Barnes-Hut. Per-component self-gravity.
      Softening lengths. Supported Methods

   .. grid-item-card:: External Conservative Force Classes
      :link: external_conservative_forces_and_potentials
      :link-type: doc

      Define and apply conservative forces from potentials.

   .. grid-item-card:: Usage as a Standalone Force Calculator
      :link: standalone_force_usage
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
   :maxdepth: 2

   setting_up_a_simulation
   generating_initial_conditions
   adding_particles_to_a_sim
   adding_external_forces_to_a_sim
   running_a_simulation
   integrators
   simulation_outputs
   evaluating_properties_of_multiple_sets_of_particles

.. toctree::
   :hidden:
   :caption: Forces
   :maxdepth: 2

   force_types
   self_gravity_force_and_solvers
   external_conservative_forces_and_potentials
   standalone_force_usage

.. toctree::
   :hidden:
   :caption: Tools & Interoperability
   :maxdepth: 1

   satellite_tools
   interoperability
   units
