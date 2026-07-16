tambora License
===============

tambora is Copyright (c) 2026 Gabriel Pfaffman.

tambora is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

tambora is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/gpl-2.0.html>.

The full license text is in the ``LICENSE`` file at the root of this
repository.


Why GPL
-------

tambora bundles the falcON tree-code by Walter Dehnen -- the same code
distributed as ``gyrfalcON`` in NEMO -- under::

    tambora/dynamics/forces/self_gravity/falcON/_falcON_src/

Those sources are Copyright (C) Walter Dehnen and are licensed under the GNU
General Public License, version 2 or (at your option) any later version.  Two
files carry additional copyright: ``inc/utils/timer.h`` (Song Ho Ahn, Walter
Dehnen) and ``inc/public/simd.h`` (Walter Dehnen, Paul McMillan).

``setup.py`` compiles those sources directly into the ``_falcon`` extension
module, which is shipped inside the ``tambora`` package.  tambora is therefore a
work based on falcON, and the GPL requires that the whole be distributed under
the same terms.  This is not a preference; it follows from bundling falcON.


Bundled and derived works
-------------------------

falcON
    ``tambora/dynamics/forces/self_gravity/falcON/_falcON_src/``, Copyright (C)
    Walter Dehnen and others; GPL-2.0-or-later.  Upstream:
    https://td.mpia.de/~dehnen/falcON/  Each file retains its original license
    header, as the GPL requires.

OpenAstronomy packaging guide
    This package's packaging scaffolding is based on the OpenAstronomy packaging
    guide (https://github.com/OpenAstronomy/packaging-guide), which is licensed
    under the BSD 3-Clause license.  See ``TEMPLATE_LICENSE.rst`` in this folder.
    BSD 3-Clause is GPL-compatible in this direction, so that material may be
    included in this GPL-licensed work.
