'''
Leapfrog Integrator.

'''

def leapfrog_kick(pos, vel, acc, dt):
    """Perform a single leapfrog step."""
    # Kick: update velocity by half step
    vel_half = vel + acc * dt
    return pos, vel_half

def leapfrog_drift(pos, vel, dt):
    """Perform the drift step of leapfrog."""
    # Drift: update position by full step using half-step velocity
    pos_new = pos + vel * dt
    return pos_new, vel