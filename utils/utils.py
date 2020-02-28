import numpy as np
import pdb

def ecef2lla(p_ECEF):
    """ Converts ECEF coordinates (x, y, z) (meters) to the
    LLA (longitude, latitude, altitude) coordinates in degrees/meters.
    Uses WGS84 geodetic coordinates and computes the 
    latitude using Bowring's method. """
    x = p_ECEF[0]
    y = p_ECEF[1]
    z = p_ECEF[2]

    if x == 0.0 and y == 0.0 and z == 0.0:
        return np.nan, np.nan, np.nan

    a = 6378137
    f = 1/298.257223563
    e2 = 2*f - f**2 # eccentricity squared

    # Initialize using spherical coordinates
    s = np.sqrt(x**2 + y**2)
    beta = np.arctan(z/((1-f)*s))
    lat = np.arctan((z + a*np.sin(beta)**3*(e2*(1-f)/(1-e2)))/(s - e2*a*np.cos(beta)**3))
    for i in range(10):
        lat_prev = lat
        beta = np.arctan((1-f)*np.sin(lat)/(np.cos(lat)))
        lat = np.arctan((z + a*np.sin(beta)**3*(e2*(1-f)/(1-e2)))/(s - e2*a*np.cos(beta)**3))
        if np.abs(lat_prev - lat) < 1e-6:
            break

    # Altitude
    rn = a/(np.sqrt(1 - e2*np.sin(lat)**2))
    h = s*np.cos(lat) + (z + e2*rn*np.sin(lat))*np.sin(lat) - rn
    lat = np.rad2deg(lat)
    lon = np.rad2deg(np.arctan2(y, x))

    return np.array([lat, lon, h])


def lla2ecef(p_LLA):
    """ Converts LLA (lat, lon, h) coordinates in degrees/meters
    to the ECEF coordinates in meters. """
    lat = p_LLA[0]
    lon = p_LLA[1]
    h = p_LLA[2]

    a = 6378137 # meters
    finv = 298.257223563
    e2 = 2*(1/finv) - (1/finv)**2 # eccentricity squared
    rn = a/(np.sqrt(1 - e2*np.sin(np.deg2rad(lat))**2))
    x_E = (rn + h)*np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(lon))
    y_E = (rn + h)*np.cos(np.deg2rad(lat))*np.sin(np.deg2rad(lon))
    z_E = (rn*(1 - e2) + h)*np.sin(np.deg2rad(lat))

    return np.array([x_E, y_E, z_E])


def ecef2enu(p_ECEF, p_ref_ECEF):
    """ Transforms the position p_ECEF in ECEF coordinates to the 
    position p_ENU that is written with respect to an ENU coordinate
    frame at position p_ref_ECEF in ECEF coordinates. """

    # Compute reference position in LLA coordinates
    p_LLA = ecef2lla(p_ref_ECEF)
    lat = np.deg2rad(p_LLA[0])
    lon = np.deg2rad(p_LLA[1])

    R = np.zeros((3,3))
    R[0,0] = -np.sin(lon)
    R[0,1] = np.cos(lon)
    R[0,2] = 0.0
    R[1,0] = -np.sin(lat)*np.cos(lon)
    R[1,1] = -np.sin(lat)*np.sin(lon)
    R[1,2] = np.cos(lat)
    R[2,0] = np.cos(lat)*np.cos(lon)
    R[2,1] = np.cos(lat)*np.sin(lon)
    R[2,2] = np.sin(lat)

    return np.matmul(R, p_ECEF - p_ref_ECEF)


def enu2ecef(p_ENU, p_ref_ECEF):
    # Compute reference position in LLA coordinates
    p_LLA = ecef2lla(p_ref_ECEF)
    lat = np.deg2rad(p_LLA[0])
    lon = np.deg2rad(p_LLA[1])

    R = np.zeros((3,3))
    R[0,0] = -np.sin(lon)
    R[0,1] = -np.sin(lat)*np.cos(lon)
    R[0,2] = np.cos(lat)*np.cos(lon)
    R[1,0] = np.cos(lon)
    R[1,1] = -np.sin(lat)*np.sin(lon)
    R[1,2] = np.cos(lat)*np.sin(lon)
    R[2,0] = 0.0
    R[2,1] = np.cos(lat)
    R[2,2] = np.sin(lat)

    return np.matmul(R, p_ENU) + p_ref_ECEF



if __name__ == '__main__':
    x = -2.7002e6
    y = -4.2928e6
    z = 3.8550e6
    lat, lon, h = ecef2lla(x,y,z)
    x_E, y_E, z_E = lla2ecef(lat, lon, h)

    
    x_ref, y_ref, z_ref = lla2ecef(37.4419, -122.1430, 0)

    p_ECEF = np.array([x, y, z])
    p_ref_ECEF = np.array([x_ref, y_ref, z_ref])
    p_ENU = ecef2enu(p_ECEF, p_ref_ECEF)

    pdb.set_trace()