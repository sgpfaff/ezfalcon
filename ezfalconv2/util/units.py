'''
Unit Conversions for ezfalcon.

Input/Output units (for user):
-----------------------------
- positions: kpc
- velocities: km/s
- masses: kg
- time: Myr
- potential : km²/s²
- acceleration: km/s²

Internal units (for simulation):
-----------------------------
- positions: kpc
- velocities: kpc/Myr
- masses: Msun
- time: Myr
- potential : (kpc/Myr)²
- acceleration: kpc/Myr²
'''


#: Gravitational constant [kpc³ Msun⁻¹ Myr⁻²]
G_INTERNAL = 4.498502151575286e-12

#: G in mixed units [kpc (km/s)² Msun⁻¹]  (used by agama & galpy bridges)
G_KPC_KMS = 4.3009172706e-06

KM_TO_KPC = 3.2407792894443656e-17
#: 1 km/s in kpc/Myr
KMS_TO_KPCMYR = 0.001022712165045695

#: 1 kpc/Myr in km/s  (≈ 977.8)
KPCMYR_TO_KMS = 1.0 / KMS_TO_KPCMYR

#: 1 Gyr in Myr
GYR_TO_MYR = 1000.0