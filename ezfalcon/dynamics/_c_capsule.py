"""Helpers for working with ezfalcon C ABI PyCapsules from Python."""

import ctypes

_FORCE_CAPSULE_NAME      = b"ezfalcon_force_v1"
_INTEGRATOR_CAPSULE_NAME = b"ezfalcon_integrator_v1"

_pycapsule = ctypes.pythonapi

_pycapsule.PyCapsule_IsValid.restype  = ctypes.c_int
_pycapsule.PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]

_pycapsule.PyCapsule_GetPointer.restype  = ctypes.c_void_p
_pycapsule.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def is_force_capsule(capsule) -> bool:
    return bool(_pycapsule.PyCapsule_IsValid(capsule, _FORCE_CAPSULE_NAME))


def is_integrator_capsule(capsule) -> bool:
    return bool(_pycapsule.PyCapsule_IsValid(capsule, _INTEGRATOR_CAPSULE_NAME))


def get_force_ptr(capsule) -> int:
    """Return the raw pointer value from a force capsule (for debugging / testing)."""
    if not is_force_capsule(capsule):
        raise TypeError("Expected an ezfalcon_force_v1 capsule.")
    return _pycapsule.PyCapsule_GetPointer(capsule, _FORCE_CAPSULE_NAME)


def get_integrator_ptr(capsule) -> int:
    """Return the raw pointer value from an integrator capsule."""
    if not is_integrator_capsule(capsule):
        raise TypeError("Expected an ezfalcon_integrator_v1 capsule.")
    return _pycapsule.PyCapsule_GetPointer(capsule, _INTEGRATOR_CAPSULE_NAME)
