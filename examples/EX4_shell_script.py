
from galpy.df import isotropicPlummerdf
from galpy.potential import PlummerPotential, NFWPotential
from ezfalcon.simulation import Sim
from ezfalcon.util import galpydfsampler
import astropy.units as u
import numpy as np



def create_shell(n, df, host_pot):
    shell = Sim()
    pos, vel, mass = galpydfsampler(df=df, n=n, m_total=1e9, center_pos=[25, 0, 0], center_vel=[0, 0, 0])
    shell.add_particles('all', pos=pos, vel=vel, mass=mass)
    shell.add_external_pot('host', host_pot)
    return shell

def create_prog_orbit(shell, host_pot):
    init_center_pos, init_center_vel = np.median(shell.pos(0), axis=0), np.median(shell.vel(0), axis=0)
    prog = Sim()
    prog.add_particles('prog', pos=init_center_pos[None,:], vel=init_center_vel[None,:], mass=np.array([1e9]))
    prog.add_external_pot('host', host_pot)
    prog.turn_self_gravity_off()
    return prog



def calculate_bound(pos, vel, mass, self_pot, center_pos, center_vel):
    '''
    Determine which stars are bound through iterative 
    calculation.
    '''
    f_xv = np.hstack((pos, vel))
    f_center = np.hstack((center_pos, center_vel))
    Rmax = 10.0
    use  = np.sum((pos - center_pos)**2, axis=1) < Rmax**2
    # iteratively refine the selection, retaining only bound particles (which have
    # negative total energy in the satellite-centered frame using its own potential only)
    prev_f_center = f_center
    for i in range(50):
        f_center = np.median(f_xv[use], axis=0)
        f_bound = self_pot + 0.5 * mass* np.sum((f_xv[:,3:6] - f_center[3:6])**2, axis=1) < 0
        if np.sum(f_bound)<=1 or all(f_center==prev_f_center): break
        use = f_bound * (np.sum((f_xv[:,0:3] - f_center[0:3])**2, axis=1) < Rmax**2)
    return f_bound

def calculate_star_mask(vel, mass, self_PE):
    center_vel = np.mean(vel, axis=0)
    binding = self_PE + 0.5 * mass* np.sum((vel - center_vel)**2, axis=1)
    sorted_binding_indices = np.argsort(binding)
    return sorted_binding_indices[:int(0.1*len(binding))]
    
def __main__(n=1000, t_end = 100, dt=1., dt_out=10, eps=0.01, theta=0.001):
    host_pot = NFWPotential(amp=1e12 * u.Msun, a=20*u.kpc)
    sat_pot = PlummerPotential(amp=1e9 * u.Msun, b=1*u.kpc)
    df = isotropicPlummerdf(pot = sat_pot)
    shell = create_shell(n=n, df=df, host_pot=host_pot)
    prog = create_prog_orbit(shell, host_pot)

    shell.run(t_end=t_end, dt=dt, dt_out=dt_out, eps=eps, theta=theta)
    prog.run(t_end=t_end, dt=dt, dt_out=dt_out, eps=eps, theta=theta)

    mask = []
    for i, t in enumerate(shell.times):
        mask.append(calculate_bound(shell.pos(i), shell.vel(i), shell.mass, shell.self_PE(i), prog.pos(i)[0], prog.vel(i)[0]))
        print(f'{np.sum(mask[i])} bound particles at time {t:.1f}')

    star_mask = calculate_star_mask(shell.vel(0), shell.mass, shell.self_PE(0))

    np.save('script_output/bound_mask.npy', mask)
    np.save('script_output/shell_pos.npy', shell.pos())
    np.save('script_output/shell_vel.npy', shell.vel())
    np.save('script_output/shell_mass.npy', shell.mass)
    np.save('script_output/times.npy', shell.times)
    np.save('script_output/self_PE.npy', shell.self_PE())
    np.save('script_output/star_mask.npy', star_mask)


if __name__ == '__main__':
    __main__()



