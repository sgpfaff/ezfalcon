'''
Leapfrog Integrator.

'''

def leapfrog_kick(vel, acc, dt):
    """Perform a single leapfrog step."""
    # Kick: update velocity by half step
    return vel + acc * dt

def leapfrog_drift(pos, vel, dt):
    """Perform the drift step of leapfrog."""
    # Drift: update position by full step using half-step velocity
    return pos + vel * dt