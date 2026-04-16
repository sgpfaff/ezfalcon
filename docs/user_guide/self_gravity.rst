Self-Gravity
============

🚧 *Still working on it...*


falcON
------

🚧 *Still working on it...*

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Parameter
     - Default
     - Description
   * - ``eps``
     - *required*
     - Gravitational softening length in kpc
   * - ``theta``
     - 0.6
     - Tree opening angle. Smaller values give higher accuracy at increased
       computational cost.
   * - ``kernel``
     - 1
     - Softening kernel. 0 = Plummer, 1 = default (~r⁻⁷ decay),
       2–3 = faster decay kernels.


Direct Summation
----------------

🚧 *Still working on it...*

Two implementations are available: ``'direct'`` (pure Python) and
``'direct_C'`` (C extension). Both produce identical results.

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Parameter
     - Default
     - Description
   * - ``eps``
     - *required*
     - Gravitational softening length in kpc. Can be a scalar or an (N,)
       array for per-particle softening (pairwise softening uses the
       arithmetic mean).


API
---

.. autofunction:: ezfalcon.dynamics.acceleration.self_gravity
