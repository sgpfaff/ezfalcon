
# tambora

*An N-body code for the modern era.*

[![Tests](https://github.com/sgpfaff/tambora/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/sgpfaff/tambora/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/sgpfaff/tambora/branch/master/graph/badge.svg)](https://codecov.io/gh/sgpfaff/tambora)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey)](https://github.com/sgpfaff/tambora/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/sgpfaff/tambora/actions/workflows/tests.yml)
[![galpy](https://img.shields.io/badge/galpy-1.9%20%7C%201.10%20%7C%201.11%20%7C%201.12-blue)](https://github.com/sgpfaff/tambora/actions/workflows/tests.yml)


## Installation
🚧🚨 **Warning!** 🚧🚨 [tambora](https://tambora.readthedocs.io/en/latest/) is still under development. Backwards compatibility is currently not guaranteed.


### Basic Installation
The latest pre-release version of [tambora](https://tambora.readthedocs.io/en/latest/) can be installed using pip as follows:

````bash
pip install tambora==0.1.0a1.dev17
````

### Optional Dependencies


[galpy](https://docs.galpy.org/en/stable/) is required to be installed in the same environment as [tambora](https://tambora.readthedocs.io/en/latest/) to use features including galpy external potentials and distribution function sampling. Please refer to the [galpy installation guide](https://docs.galpy.org/en/stable/installation.html) for installing galpy.



## License

This project is Copyright (c) Gabriel Pfaffman and is licensed under the terms of the
[GNU General Public License v2.0 or later](LICENSE) (`GPL-2.0-or-later`).

tambora bundles [falcON](https://td.mpia.de/~dehnen/falcON/), the fast tree-code by Walter
Dehnen, the same code distributed as `gyrfalcON` in [NEMO](https://astronemo.readthedocs.io/)
and compiles it into the `_falcon` extension module. falcON is GPL-2.0-or-later, so tambora as
distributed is a combined work and carries the same licence. See
[licenses/LICENSE.rst](licenses/LICENSE.rst) for the full provenance.

If you use the falcON self-gravity backend, please cite Dehnen (2000, 2002) alongside tambora.

This package's packaging scaffolding is based upon the
[OpenAstronomy packaging guide](https://github.com/OpenAstronomy/packaging-guide), which is
licensed under the BSD 3-Clause licence. See the licenses folder for more information.

## Contributing

We love contributions! tambora is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
tambora based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.


