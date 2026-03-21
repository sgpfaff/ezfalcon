import numpy as np

def cyl2cart(R, phi, z):
    '''
    Convert cylindrical coordinates to Cartesian coordinates.

    Parameters
    ----------
    R : array-like
        Cylindrical radius (kpc)
    vR : array-like
        Radial velocity (kpc/Myr)
    vT : array-like
        Tangential velocity (kpc/Myr)
    z : array-like
        Height above the plane (kpc)
    vz : array-like
        Vertical velocity (kpc/Myr)
    phi : array-like
        Azimuthal angle (radians)

    Returns
    -------
    x : array-like
        x-coordinate in Cartesian coordinates (kpc)
    y : array-like
        y-coordinate in Cartesian coordinates (kpc)
    z : array-like
        z-coordinate in Cartesian coordinates (kpc)
    vx : array-like
        x-component of velocity in Cartesian coordinates (kpc/Myr)
    vy : array-like
        y-component of velocity in Cartesian coordinates (kpc/Myr)
    vz : array-like
        z-component of velocity in Cartesian coordinates (kpc/Myr)
    '''
    x = R * np.cos(phi)
    y = R * np.sin(phi)



    return x, y, z

def cyl2cart_vec(vR, vT, phi):
    '''
    Convert cylindrical vector components to Cartesian components.
    Parameters
    ----------
    vR : array-like
        Radial component
    vT : array-like
        Tangential component 
    vz : array-like
        Azimuthal angle

    Returns
    -------
    vx : array-like
        x-component in Cartesian coordinates
    vy : array-like
        y-component in Cartesian coordinates 

    '''
    vx = vR * np.cos(phi) - vT * np.sin(phi)
    vy = vR * np.sin(phi) + vT * np.cos(phi)