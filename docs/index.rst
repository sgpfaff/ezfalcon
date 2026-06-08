.. _home:

tambora
========


.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      How to install tambora and its optional dependencies.

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc

      A minimal working example to get up and running fast.

   .. grid-item-card:: User Guide
      :link: user_guide/index
      :link-type: doc
      :img-bottom: user_guide/multicomponent_IC.png

      In-depth explanations of tambora's core concepts: units, self-gravity,
      external forces, and more.

   .. grid-item-card:: Examples
      :link: examples/index
      :link-type: doc
      :img-bottom: examples/GC_stream_evolution.gif

      Notebooks covering tambora's features, from basics to
      science applications.

   .. .. grid-item-card:: Diagnostics
   ..    :link: diagnostics/index
   ..    :link-type: doc

   ..    Convergence tests, scaling benchmarks, and energy conservation
   ..    validation.

   .. .. grid-item-card:: What's New
   ..    :link: changelog
   ..    :link-type: doc

      Release notes and the changelog for each version of tambora.

.. The navbar is built from the top-level toctree below (Installation,
.. Quickstart, User Guide, Examples). Diagnostics and the changelog are kept
.. off the navbar on purpose — they are reachable from the cards above and are
.. marked :orphan: so they still build and remain linkable.

.. toctree::
   :hidden:

   installation
   quickstart
   user_guide/index
   examples/index
