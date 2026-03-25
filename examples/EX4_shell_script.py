
from galpy.df import isotropicPlummerdf
from galpy.potential import PlummerPotential, NFWPotential
from ezfalcon.simulation import Sim
from ezfalcon.util import galpydfsampler
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'font.size': 14, 'axes.labelsize': 16,
    'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'ytick.right': True, 'xtick.top': True,
    'mathtext.fontset': 'stix',
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
})

from matplotlib.animation import FuncAnimation
from IPython.display import HTML



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
    

def make_animation(x, y, times, filename, mask=None, label1='', label2=''):
    fig, ax = plt.subplots(figsize=(6, 6))

    def animate(i):
        ax.clear()
        if mask is not None:
            ax.scatter(x[i][~mask[i]], y[i][~mask[i]], s=2, c='k', alpha=0.25, label=label1)
            ax.scatter(x[i][mask[i]], y[i][mask[i]], s=1, c='k', label=label2)
            ax.legend(loc='upper right', markerscale=5)
        else:
            ax.scatter(x[i], y[i], s=1, c='k')

        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.set_xlabel('x [kpc]')
        ax.set_ylabel('y [kpc]')
        ax.set_title(f't = {times[i]:.0f} Myr')
        

    anim = FuncAnimation(fig, animate, frames=len(times), interval=100)
    anim.save(f'script_output/{filename}.mp4', writer='ffmpeg', fps=2)
    plt.close(fig)

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

    np.save('script_output/data/bound_mask.npy', mask)
    np.save('script_output/data/shell_pos.npy', shell.pos())
    np.save('script_output/data/shell_vel.npy', shell.vel())
    np.save('script_output/data/shell_mass.npy', shell.mass)
    np.save('script_output/data/times.npy', shell.times)
    np.save('script_output/data/self_PE.npy', shell.self_PE())
    np.save('script_output/data/star_mask.npy', star_mask)

    shell.plot_diagnostic('script_output/energy_conservation.png')
    make_animation(shell.pos()[:,:,0], shell.pos()[:,:,1], shell.times, 'anims/all_pts')
    make_animation(shell.pos()[:,:,0], shell.pos()[:,:,1], shell.times, 'anims/bound_unbound', mask=mask, label1='unbound', label2='bound')
    make_animation(shell.pos()[:,:,0], shell.pos()[:,:,1], shell.times, 'anims/stars', mask=star_mask, label1='DM', label2='stars')
    make_animation(shell.pos()[:,star_mask,0], shell.pos()[:,star_mask,1], shell.times, 'anims/stars_only')

if __name__ == '__main__':
    __main__()



