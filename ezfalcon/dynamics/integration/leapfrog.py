'''
Leapfrog Integrator.

'''

def leapfrog_step(pos, vel, acc_fn, dt):
    """Perform a single leapfrog step."""
    # Drift-Kick-Drift sequence
    pos_half = leapfrog_drift(pos, vel, dt/2)
    acc, _, _ = acc_fn(pos_half)
    vel_new = leapfrog_kick(vel, acc, dt)
    pos_new = leapfrog_drift(pos_half, vel_new, dt/2)
    return pos_new, vel_new

def leapfrog_kick(vel, acc, dt):
    """Perform a single leapfrog step."""
    # Kick: update velocity by half step
    return vel + acc * dt

def leapfrog_drift(pos, vel, dt):
    """Perform the drift step of leapfrog."""
    # Drift: update position by full step using half-step velocity
    return pos + vel * dt