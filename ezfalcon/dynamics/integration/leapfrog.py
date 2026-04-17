'''
Leapfrog Integrator.

'''

def _leapfrog_step(pos, vel, acc0, acc_fn, dt, t=None):
    """Perform a single leapfrog step."""
    # Kick-Drift-Kick sequence
    vel_half = leapfrog_kick(vel, acc0, dt/2)
    pos = leapfrog_drift(pos, vel_half, dt)
    acc, self_gravity, self_pot = acc_fn(pos, t=t)
    vel = leapfrog_kick(vel_half, acc, dt/2)

    return pos, vel, acc, self_gravity, self_pot

def leapfrog_kick(vel, acc, dt):
    """Perform a single leapfrog step."""
    # Kick: update velocity by half step
    return vel + acc * dt

def leapfrog_drift(pos, vel, dt):
    """Perform the drift step of leapfrog."""
    # Drift: update position by full step using half-step velocity
    return pos + vel * dt