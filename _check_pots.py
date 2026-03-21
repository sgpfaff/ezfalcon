import numpy as np, warnings, astropy.units as u
from functools import partial
from ezfalconv2.util import galpy as galpy_util
from galpy.potential import (
    MiyamotoNagaiPotential, LogarithmicHaloPotential,
    FlattenedPowerPotential, MN3ExponentialDiskPotential,
    KuzminKutuzovStaeckelPotential, RingPotential, SCFPotential,
    TriaxialHernquistPotential, TriaxialNFWPotential, TriaxialJaffePotential,
    TwoPowerTriaxialPotential, PerfectEllipsoidPotential,
    FerrersPotential, DehnenBarPotential, SoftenedNeedleBarPotential,
    SpiralArmsPotential,
)

def numerical_acc(pot_fn, pos, h=1e-5):
    pos = np.asarray(pos, dtype=float)
    acc = np.zeros_like(pos)
    for i in range(3):
        pp = pos.copy(); pm = pos.copy()
        pp[:, i] += h; pm[:, i] -= h
        acc[:, i] = -(pot_fn(pp) - pot_fn(pm)) / (2 * h)
    return acc

r_sweep = np.geomspace(1e-14, 100, 500)
diag = r_sweep / np.sqrt(3)
pos_sweep = np.column_stack([diag, diag, diag])
threshold = 1e-6

axi = {
    'MiyamotoNagai': MiyamotoNagaiPotential(),
    'LogHalo(q=0.8)': LogarithmicHaloPotential(q=0.8),
    'FlattenedPower': FlattenedPowerPotential(),
    'MN3ExpDisk': MN3ExponentialDiskPotential(),
    'KuzminKutuzov': KuzminKutuzovStaeckelPotential(),
    'Ring': RingPotential(),
    'SCF': SCFPotential(),
}
tri = {
    'TriaxialHernquist': TriaxialHernquistPotential(b=0.8, c=0.6),
    'TriaxialNFW': TriaxialNFWPotential(b=0.8, c=0.6),
    'TriaxialJaffe': TriaxialJaffePotential(b=0.8, c=0.6),
    'TwoPowerTriaxial': TwoPowerTriaxialPotential(b=0.8, c=0.6),
    'PerfectEllipsoid': PerfectEllipsoidPotential(b=0.8, c=0.6),
    'Ferrers': FerrersPotential(),
    'DehnenBar': DehnenBarPotential(),
    'SoftenedNeedleBar': SoftenedNeedleBarPotential(),
    'SpiralArms': SpiralArmsPotential(),
    'LogHalo(triax)': LogarithmicHaloPotential(b=0.8, q=0.7),
}

for group_name, pots in [('AXISYMMETRIC', axi), ('TRIAXIAL', tri)]:
    print(f'\n=== {group_name} ===')
    print(f"{'Potential':<25} {'r_crit':>14} {'r_nan':>10}")
    print('-'*51)
    for p in pots.values():
        p.turn_physical_on()
    for name, pot in sorted(pots.items()):
        acc_fn = galpy_util._galpy_pot_to_acc_fn(pot)
        pot_fn = galpy_util._galpy_pot_to_pot_fn(pot)
        pot_i = partial(pot_fn, t=0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ab = acc_fn(pos_sweep, t=0)
                an = numerical_acc(pot_i, pos_sweep)
        except Exception as e:
            print(f'{name:<25} SKIPPED ({type(e).__name__})')
            continue
        mb = np.linalg.norm(ab, axis=1)
        mn = np.linalg.norm(an, axis=1)
        good = mn > 1e-30
        re = np.full_like(r_sweep, np.nan)
        re[good] = np.abs(mb[good]-mn[good])/mn[good]
        bad = np.where(good & (re > threshold))[0]
        rc = f'{r_sweep[bad.max()]:.2e}' if len(bad) > 0 else 'always < 1e-6'
        nm = np.isnan(ab).any(axis=1)
        rn = f'{r_sweep[nm].max():.2e}' if nm.any() else 'None'
        print(f'{name:<25} {rc:>14} {rn:>10}')
