from ..forces import SelfGravityForce, Conservative, Force
from ..operations.step_context import StepContext, base_providers
from ..operations.diagnostic import Diagnostic
from ..operations.accumulator import Accumulator
from .BaseIntegrator import BaseIntegrator
import numpy as np
from tqdm import tqdm
from functools import partial
from typing import Optional
import warnings

def _runner(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, 
            integrator: BaseIntegrator, 
            self_gravity_force: Optional[SelfGravityForce], 
            conserv_ext_force: Optional[Conservative],
            base_ext_force: Optional[Force], t0: float,
            t_end: float, dt: float, dt_out: float,
            return_self_gravity_pot: bool = True,
            return_self_gravity_acc: bool = True,
            operations=(), record_into=None, summaries_into=None):
    '''
    Integrate particle trajectories optionally under 
    influence of self-gravity and external forces.

    Parameters
    ----------
    pos : ndarray
        The starting positions of particles.
    vel : ndarray
        The starting velocities of particles.
    mass : ndarray
        The mass of particles.
    integrator: BaseIntegrator
        Integrator class to use, inherited from BaseIntegrator.
    self_gravity_force : SelfGravity
        self-gravity solver class.
    conserv_ext_forces: Conservative
        Conservative external forces.
    base_ext_forces : Force
        Non-conservative external forces.
    t0 : float
        Start time of integration. Default is 0.0 Gyr.
        Units: `Gyr`
    t_end : float
        End time of integration.
        Units: `Gyr`
    dt : float
        Timestep for integration.
        Units: `Gyr`
    dt_out : float
        Output interval.
        Units: `Gyr`
    return_self_gravity_pot : bool, optional
        Whether to return the self-gravitational potential at each output snapshot. Default is True.
    return_self_gravity_acc : bool, optional
        Whether to return the self-gravitational acceleration at each output snapshot. Default is True.
    **kwargs
        Additional keyword arguments to pass to the self-gravity method.

    Returns
    -------
    positions : (nsnaps, N, 3) array
        Positions at each output snapshot.
        Units: `kpc`
    velocities : (nsnaps, N, 3) array
        Velocities at each output snapshot.
        Units: `kpc / Gyr`
    ts_out : (nsnaps,) array
        Times of each output snapshot.
        Units: `Gyr`
    self_gravity_acc : (nsnaps, N, 3) array or None
        Self-gravitational accelerations at each output snapshot. 
        Returns None if return_self_gravity is False.
        Units: `kpc / Gyr^2`
    self_gravity_pot : (nsnaps, N) array or None
        Self-gravitational potentials at each output snapshot.
        Returns None if return_self_potential is False.
        Units: `kpc^2 / Myr^2`
    '''
    _check_dt_dt_out(dt, dt_out, t0, t_end)

    (ts_out, ts_integrate, 
    nsnaps, steps_per_output) = _make_time_arrays(dt, dt_out, t0, t_end)

    positions, velocities = _make_pos_vel_arrays(pos, vel, mass, nsnaps)

    # --- operations: providers, record buffers, per-run state, initial snapshot ---------
    # Self-gravity acc/pot are recorded through the same generic diagnostic path as user
    # diagnostics, pulled via base providers (StepResult fast path during the run, the
    # force on the initial snapshot). The return_self_gravity_* flags just gate whether
    # those two names are recorded.
    providers = base_providers(self_gravity_force)
    diagnostics = [op for op in operations if isinstance(op, Diagnostic) and op.name]
    accumulators = [op for op in operations if isinstance(op, Accumulator)]
    displays = [op for op in operations if getattr(op, "display", None) is not None]
    providers.update({d.name: d.compute for d in diagnostics})
    sg_names = ([] + (["self_acc"] if return_self_gravity_acc else [])
                   + (["self_pot"] if return_self_gravity_pot else []))
    records = {}
    n = mass.shape[0]
    for op in operations:
        op.init(n)
    if diagnostics or sg_names:
        ctx0 = StepContext(pos, vel, mass, t0, 0, None, providers)
        for name in sg_names:
            _record_diagnostic(records, name, ctx0.get(name), 0, nsnaps)
        for d in diagnostics:
            _record_diagnostic(records, d.name, ctx0.get(d.name), 0, nsnaps)

    i_out = 1
    current_pos, current_vel = pos, vel
    current_t = t0
    postfix = {}
    integrator.reset()
    pbar = tqdm(ts_integrate[1:])
    for step, t in enumerate(pbar, start=1):
        step_result = integrator.step(current_pos, current_vel, mass, current_t, dt,
                                      self_gravity_force, conserv_ext_force, base_ext_force)
        current_pos, current_vel, current_t = step_result.pos, step_result.vel, step_result.t

        is_output = step % steps_per_output == 0 and i_out < nsnaps
        # One context per step, shared by accumulators, displays, and diagnostics so a
        # pulled quantity (e.g. boundedness) is computed at most once this step.
        ctx = None
        def _ctx():
            nonlocal ctx
            if ctx is None:
                ctx = StepContext(current_pos, current_vel, mass, current_t, step, step_result, providers)
            return ctx

        if accumulators:
            for acc in accumulators:
                if step % acc.cadence == 0:
                    acc.update(_ctx())

        if displays:
            updated = False
            for op in displays:
                if step % (op.display.every or steps_per_output) == 0:
                    label, value = op.display.text(_ctx(), op)
                    postfix[label] = value
                    updated = True
            if updated:
                pbar.set_postfix(postfix, refresh=False)

        if is_output: # recording snapshot
            positions[i_out] = step_result.pos.copy()
            velocities[i_out] = step_result.vel.copy()
            for name in sg_names:
                _record_diagnostic(records, name, _ctx().get(name), i_out, nsnaps)
            for d in diagnostics:
                if step % d.cadence == 0:
                    _record_diagnostic(records, d.name, _ctx().get(d.name), i_out, nsnaps)
            i_out += 1

    summaries = {}
    for op in operations:
        result = op.finalize()
        if result:
            summaries.update(result)

    if record_into is not None:
        record_into.update(records)
    if summaries_into is not None:
        summaries_into.update(summaries)
    return (positions, velocities, ts_out,
            records.get("self_acc"), records.get("self_pot"))


def _record_diagnostic(records, name, value, i, nsnaps):
    """Lazily allocate a ``(nsnaps, *value.shape)`` buffer and store this snapshot."""
    value = np.asarray(value)
    if name not in records:
        records[name] = np.empty((nsnaps,) + value.shape, dtype=value.dtype)
    records[name][i] = value


def _check_dt_dt_out(dt, dt_out, t0, t_end):
    if (t0 > t_end) and (dt > 0):
        raise ValueError("The end time (t_end) is less than the start time (t0), implying backwards integration. " \
        "dt must be negative for backwards integration.")
    elif (t0 < t_end) and (dt < 0):
        raise ValueError("The end time (t_end) is greater than the start time (t0), implying forwards integration. " \
        "dt must be positive for forwards integration.")
    if np.abs(dt_out) < np.abs(dt):
        raise ValueError("The absolute value of dt_out must be greater than or equal to dt.")
    if np.sign(dt_out) != np.sign(dt):
        raise ValueError("dt_out must have the same sign as dt.")
    if abs(dt_out / dt - round(dt_out / dt)) > 1e-9:
        raise ValueError("dt_out must be a multiple of dt.")
    if abs((t_end - t0) / dt - round((t_end  - t0) / dt)) > 1e-9:
        actual_t_end = int((t_end - t0) / dt) * dt + t0
        warnings.warn(f"Simulation duration ({t_end - t0} Gyr) is not an exact multiple of dt={dt} Gyr. "
                        f"The simulation will end before t_end it reaches t_end.")
    if abs((t_end - t0) / dt_out - round((t_end - t0) / dt_out)) > 1e-9:
        n_steps_w = int((t_end - t0) / dt) if abs((t_end - t0)/ dt - round((t_end - t0) / dt)) > 1e-9 else round((t_end - t0) / dt)
        steps_per_output = round(dt_out / dt)
        nsnaps = n_steps_w // steps_per_output
        actual_t_end = nsnaps * dt_out + t0
        warnings.warn(f"Simulation duration ({t_end - t0} Gyr) is not an exact multiple of dt_out={dt_out} Gyr. "
                        f"Last output will be at t={actual_t_end:.10g} Gyr instead of t={t_end} Gyr.")

def _make_time_arrays(dt, dt_out, t0, t_end):
    ratio_save = (t_end - t0) / dt_out
    n_steps_save = round(ratio_save) if abs(ratio_save - round(ratio_save)) < 1e-9 else int(ratio_save)

    ratio_integrate = (t_end - t0) / dt
    n_steps_integrate = round(ratio_integrate) if abs(ratio_integrate - round(ratio_integrate)) < 1e-9 else int(ratio_integrate)

    steps_per_output = round(dt_out / dt)
    nsnaps = n_steps_save + 1  # +1 for initial snapshot at t=0
    ts_out = np.arange(nsnaps, dtype=np.float64) * dt_out + t0
    ts_integrate = np.arange(n_steps_integrate + 1, dtype=np.float64) * dt + t0
    return ts_out, ts_integrate, nsnaps, steps_per_output


def _make_pos_vel_arrays(pos, vel, mass, nsnaps):
    n = mass.shape[0]
    positions = np.zeros((nsnaps, n, 3), dtype=np.float64)
    velocities = np.zeros((nsnaps, n, 3), dtype=np.float64)
    positions[0] = pos.copy()
    velocities[0] = vel.copy()
    return positions, velocities