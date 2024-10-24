import requests
import json
# import xmltodict
# from pprint import pprint as pp

# import math
# import datetime



# headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
#         }
# # Fetch the XML file from the URL
# url = 'https://www.gsc-europa.eu/sites/default/files/sites/all/files/2024-10-15.xml'
# response = requests.get(url, headers=headers)
# # response = requests.get(url)
# text = response.text
# json_obj = xmltodict.parse(text)
# almanacs = json_obj["signalData"]["body"]["Almanacs"]["svAlmanac"]



# celestrak_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=json-pretty"
# response = requests.get(celestrak_url).json()
# print(response)

# for sat in response:

response = json.load(open("data_gps.json"))["DATA"]
satellite_params = response



# def calculate_position(satellite, time_offset):
#     # Constants
#     mu = 3.986005e14  # Earth's gravitational constant (m^3/s^2)
    
#     # Extract almanac parameters
#     almanac = satellite['almanac']
#     a = float(almanac['aSqRoot']) ** 2  # semi-major axis (m)
#     ecc = float(almanac['ecc'])  # eccentricity
#     m0 = float(almanac['m0'])  # mean anomaly at reference time (rad)
#     omega0 = float(almanac['omega0'])  # right ascension of ascending node (rad)
#     w = float(almanac['w'])  # argument of perigee (rad)
#     omegaDot = float(almanac['omegaDot'])  # rate of right ascension (rad/s)
#     t0a = float(almanac['t0a'])  # reference time (seconds from epoch)

#     # Current time and the time at which to calculate the position
#     current_time = datetime.datetime.now()
#     target_time = current_time + datetime.timedelta(hours=time_offset)
    
#     # Calculate the time difference in seconds
#     t = (target_time - current_time).total_seconds()
    
#     # Mean anomaly at time t
#     n = math.sqrt(mu / (a ** 3))  # mean motion (rad/s)
#     M = m0 + n * (t + t0a)  # mean anomaly (rad)
    
#     # Solve Kepler's equation for eccentric anomaly (E)
#     E = M  # Initial guess
#     for _ in range(10):  # Iterative solution
#         E = M + ecc * math.sin(E)
    
#     # Calculate position in the orbital plane
#     x_orbital = a * (math.cos(E) - ecc)
#     y_orbital = a * math.sqrt(1 - ecc**2) * math.sin(E)

#     # Calculate the true anomaly
#     true_anomaly = 2 * math.atan2(math.sqrt(1 + ecc) * math.sin(E / 2), math.sqrt(1 - ecc) * math.cos(E / 2))

#     # Calculate satellite's position in the inertial frame
#     x = x_orbital * math.cos(omega0) - y_orbital * math.sin(omega0) * math.cos(w)
#     y = x_orbital * math.sin(omega0) + y_orbital * math.cos(omega0) * math.cos(w)
#     z = y_orbital * math.sin(w)

#     return (x, y, z)

# # Calculate positions for the next 3 hours
# for hour in range(1, 2):
#     print(f"Positions for {hour} hour(s) from now:")
#     for satellite in almanacs:
#         position = calculate_position(satellite, hour)
#         print(f"SVID {satellite['SVID']}: Position (x, y, z) = {position}")



# import numpy as np

# # Constants
# mu = 3.986004418e14  # Earth's gravitational constant (m^3/s^2)
# R_E = 6371000  # Earth's radius in meters

# # Function to compute satellite position and DOP
# def compute_satellite_position_and_dop(satellites, observer_lat, observer_lon):
#     # Convert observer coordinates to radians
#     observer_lat_rad = np.radians(observer_lat)
#     observer_lon_rad = np.radians(observer_lon)

#     # List to hold satellite positions and DOP values
#     satellite_positions = []
#     dop_values = []

#     for satellite in satellites:
#         # Extract satellite parameters
#         aSqRoot = float(satellite['almanac']['aSqRoot'])
#         ecc = float(satellite['almanac']['ecc'])
#         m0 = float(satellite['almanac']['m0'])
#         t0a = float(satellite['almanac']['t0a'])
        
#         # Semi-major axis
#         a = aSqRoot ** 2  # Semi-major axis in meters
        
#         # Assume elapsed time (in seconds) for DOP calculation
#         elapsed_time = 3600  # 1 hour later
#         current_time = t0a + elapsed_time
        
#         # Calculate mean anomaly (M)
#         M = m0 + (elapsed_time * (np.sqrt(mu / a**3)))  # Mean motion: sqrt(mu/a^3)
#         M = M % (2 * np.pi)  # Normalize to [0, 2π]

#         # Solve for eccentric anomaly using Newton's method
#         E = M  # Initial guess
#         for _ in range(10):  # Iterate to solve for E
#             E = M + ecc * np.sin(E)

#         # Calculate true anomaly
#         v = 2 * np.arctan2(np.sqrt(1 + ecc) * np.sin(E / 2),
#                            np.sqrt(1 - ecc) * np.cos(E / 2))

#         # Calculate radius in the orbital plane
#         r = a * (1 - ecc * np.cos(E))

#         # Position in the orbital plane
#         x_prime = r * np.cos(v)
#         y_prime = r * np.sin(v)

#         # Transform to Earth-Centered Inertial coordinates (assume equatorial)
#         x = x_prime
#         y = y_prime
#         z = 0  # Neglecting altitude for simplification

#         # Save satellite position
#         satellite_positions.append((x, y, z))

#         # Compute DOP
#         d = np.sqrt((x - 0) ** 2 + (y - 0) ** 2 + (z - 0) ** 2)  # Distance from observer
#         dop = 1 / d if d != 0 else np.inf  # DOP calculation
#         dop_values.append(dop)

#     return satellite_positions, dop_values

# # Almanac data for satellites
# satellites = almanacs
# # Observer coordinates
# observer_lat = 0  # Latitude
# observer_lon = 0  # Longitude

# # Compute positions and DOP values
# positions, dop_values = compute_satellite_position_and_dop(satellites, observer_lat, observer_lon)

# # Display results
# for i, (pos, dop) in enumerate(zip(positions, dop_values)):
#     print(f"Satellite {satellites[i]['SVID']} Position: {pos}, DOP: {dop}")



# import numpy as np
# from datetime import datetime, timedelta

# Given almanac data
# almanac = {
#     "aSqRoot": 0.03515625,  # Square root of semi-major axis
#     "ecc": 0.00030517578125,  # Eccentricity
#     "deltai": -0.00372314453125063230670699354619,  # Inclination change
#     "omega0": -0.797332763671874888977697537484,  # Right ascension
#     "omegaDot": -1.86264514923107200935514487085e-09,  # Rate of right ascension
#     "w": 0.0497131347656228281262080770375,  # Argument of perigee
#     "m0": 0.0402221679687564670491184415368,  # Mean anomaly at reference time
#     "af0": 0.0001678466796875,  # Clock correction bias
#     "af1": 3.63797880709199971837642219221e-12,  # Clock correction rate
#     "t0a": 204000,  # Almanac reference time in seconds
# }

# # Current time
# t = datetime(2024, 10, 15, 23, 18, 44)  # Specify the time for position calculation
# # Calculate the number of seconds since the start of the week
# start_of_week = t - timedelta(days=t.weekday())  # Start of the week
# seconds_since_week_start = (t - start_of_week).total_seconds()

# # Mean motion (n) in rad/s
# # Semi-major axis (a) in meters: convert from aSqRoot to a
# a = (almanac["aSqRoot"] ** 2) * (29960000)  # 29,960 km nominal semi-major axis
# n = np.sqrt(3.986004418e14 / (a ** 3))  # Earth's gravitational parameter

# # Calculate mean anomaly
# M = almanac["m0"] + n * (seconds_since_week_start - almanac["t0a"])

# # Normalize M to be between 0 and 2π
# M = M % (2 * np.pi)

# # Solve for eccentric anomaly E using Kepler's equation (iteratively)
# e = almanac["ecc"]
# E = M  # Start with M
# for _ in range(10):  # Iterate to solve Kepler's equation
#     E = M + e * np.sin(E)

# # Calculate true anomaly
# nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

# # Calculate radius
# r = a * (1 - e * np.cos(E))

# # Calculate position in the orbital plane
# x_prime = r * np.cos(nu)
# y_prime = r * np.sin(nu)

# # Convert to ECEF coordinates
# omega = almanac["w"]
# omega0 = almanac["omega0"]
# deltai = almanac["deltai"]

# # ECEF coordinates
# x = x_prime * (np.cos(omega) * np.cos(omega0) - np.sin(omega) * np.sin(omega0) * np.cos(deltai))
# y = x_prime * (np.cos(omega) * np.sin(omega0) + np.sin(omega) * np.cos(omega0) * np.cos(deltai))
# z = x_prime * (np.sin(omega) * np.sin(deltai))

# # Output the satellite position
# position = (x, y, z)
# print(f"Satellite Position (ECEF): {position}")

# # Function to compute look angles
# def look_angles(receiver_pos, satellite_pos):
#     # Convert receiver position from geodetic to ECEF
#     lat, lon, alt = receiver_pos
#     # ECEF conversion constants
#     a = 6378137  # WGS-84 semi-major axis in meters
#     e2 = 0.00669437999014  # Square of first eccentricity
#     N = a / np.sqrt(1 - e2 * np.sin(np.radians(lat))**2)

#     # ECEF receiver position
#     x_r = (N + alt) * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
#     y_r = (N + alt) * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
#     z_r = (N * (1 - e2) + alt) * np.sin(np.radians(lat))

#     # Calculate the look angles
#     azimuths = []
#     elevations = []
#     for pos in satellite_pos:
#         x_s, y_s, z_s = pos
#         # Calculate the satellite position relative to the receiver
#         dx = x_s - x_r
#         dy = y_s - y_r
#         dz = z_s - z_r
#         horizontal_distance = np.sqrt(dx**2 + dy**2)

#         # Elevation angle
#         elevation = np.arctan2(dz, horizontal_distance)
#         elevation_angle = np.degrees(elevation)

#         # Azimuth angle
#         azimuth = np.arctan2(dy, dx)
#         azimuth_angle = np.degrees(azimuth) % 360

#         azimuths.append(azimuth_angle)
#         elevations.append(elevation_angle)

#     return np.array(azimuths), np.array(elevations)

# # Specify receiver position
# receiver_pos = [45.3013162, 7.6782972, 500]  # Latitude, Longitude, Altitude

# # Calculate look angles
# azimuths, elevations = look_angles(receiver_pos, [position])

# # Print results
# print(f"Azimuth: {azimuths}, Elevation: {elevations}")





import numpy as np

def solve_kepler_equation_eccentric_anomaly(e0, Mk):
    """
    Solve Kepler's equation for Eccentric Anomaly
    Mk = Ek - e0*sin(Ek)

    Inputs:
    e0 : Eccentricity
    Mk : Mean Anomaly (rad)

    Outputs:
    Ek : Eccentric Anomaly (rad)
    """
    num_iters = 22  # Empirical number to achieve millimeter accuracy.
    Ek = Mk + (e0 * np.sin(Mk)) / (1 - np.sin(Mk + e0) + np.sin(Mk))
    for _ in range(num_iters):
        Ek = Mk + e0 * np.sin(Ek)
    return Ek

def orbit_parameters_to_ecef(gps_week, t, week_num, toe, 
                             ARef, deltaA, ADot, 
                             mu, deltan0, deltan0Dot, 
                             M0, e0, omega0, 
                             i0, iDot, 
                             Cis, Cic, Crs, Crc, Cus, Cuc, 
                             OmegaEDot, OmegaRefDot, Omega0, deltaOmegaDot):
    """
    Convert orbital parameters to satellite positions and velocities in 
    Earth-Centered Earth-Fixed (ECEF) coordinates.
    """
    seconds_per_week = 604800

    # Time from ephemeris reference time
    t_sys = t
    curr_gps_week_num = gps_week
    tk = ((curr_gps_week_num - week_num) * seconds_per_week + t_sys) - toe

    # Add rollover if magnitude of |tk| is above a certain range, for GPS week crossover.
    tk = np.asarray(tk)
    tk[tk > seconds_per_week / 2] -= seconds_per_week
    tk[tk < -seconds_per_week / 2] += seconds_per_week

    # Semi-Major Axis at reference time
    A0 = ARef + deltaA
    # Semi-Major Axis
    Ak = A0 + ADot * tk

    # Computed Mean Motion
    n0 = np.sqrt(mu / A0**3)
    # Mean motion difference from computed value
    deltanA = deltan0 + 0.5 * deltan0Dot * tk
    # Corrected Mean Motion
    nA = n0 + deltanA

    # Mean Anomaly
    Mk = M0 + nA * tk
    # Eccentric Anomaly
    Ek = solve_kepler_equation_eccentric_anomaly(e0, Mk)
    cosEk = np.cos(Ek)
    sinEk = np.sin(Ek)
    # True Anomaly
    vk = np.arctan2((np.sqrt(1 - e0**2) * sinEk) / (1 - e0 * cosEk), 
                    (cosEk - e0) / (1 - e0 * cosEk))
    # Argument of Latitude
    Phik = vk + omega0
    sin2Phik = np.sin(2 * Phik)
    cos2Phik = np.cos(2 * Phik)

    # Second harmonic perturbations
    deltauk = Cus * sin2Phik + Cuc * cos2Phik
    deltark = Crs * sin2Phik + Crc * cos2Phik
    deltaik = Cis * sin2Phik + Cic * cos2Phik

    # Corrected values
    uk = Phik + deltauk
    cosuk = np.cos(uk)
    sinuk = np.sin(uk)
    rk = Ak * (1 - e0 * cosEk) + deltark
    ik = i0 + iDot * tk + deltaik
    cosik = np.cos(ik)
    sinik = np.sin(ik)

    # Positions in orbital plane
    xk_prime = rk * cosuk
    yk_prime = rk * sinuk

    # Rate of Right Ascension
    OmegaDot = OmegaRefDot + deltaOmegaDot
    # Corrected Longitude of Ascending Node
    Omegak = Omega0 + (OmegaDot - OmegaEDot) * tk - OmegaEDot * toe
    cosOmegak = np.cos(Omegak)
    sinOmegak = np.sin(Omegak)

    # ECEF Positions of Satellites
    xk = xk_prime * cosOmegak - yk_prime * cosik * sinOmegak
    yk = xk_prime * sinOmegak + yk_prime * cosik * cosOmegak
    zk = yk_prime * sinik

    # Eccentric Anomaly derivative
    EkDot = nA / (1 - e0 * cosEk)
    # Argument of Latitude derivative
    PhikDot = (np.sin(vk) / sinEk) * EkDot

    # Derivatives
    deltaukDot = 2 * (Cus * cos2Phik - Cuc * sin2Phik) * PhikDot
    deltarkDot = 2 * (Crs * cos2Phik - Crc * sin2Phik) * PhikDot
    deltaikDot = 2 * (Cis * cos2Phik - Cic * sin2Phik) * PhikDot

    # Corrected derivatives
    rkDot = (Ak * e0 * sinEk) * EkDot + deltarkDot
    ukDot = PhikDot + deltaukDot
    ikDot = iDot + deltaikDot

    # Velocities in orbital plane
    xk_primeDot = rkDot * cosuk - rk * ukDot * sinuk
    yk_primeDot = rkDot * sinuk + rk * ukDot * cosuk

    # Corrected Longitude of Ascending Node derivative
    OmegakDot = OmegaDot - OmegaEDot

    # ECEF Velocities of Satellites
    xkDot = xk_primeDot * cosOmegak - yk_primeDot * cosik * sinOmegak + ikDot * yk_prime * sinik * sinOmegak - OmegakDot * yk
    ykDot = xk_primeDot * sinOmegak + yk_primeDot * cosik * cosOmegak - ikDot * yk_prime * sinik * cosOmegak + OmegakDot * xk
    zkDot = yk_primeDot * sinik + ikDot * yk_prime * cosik

    # Concatenate outputs
    pos = np.stack([xk, yk, zk], axis=-1)
    vel = np.stack([xkDot, ykDot, zkDot], axis=-1)

    return pos, vel



from datetime import datetime, timedelta

class GalileoTime:
    @staticmethod
    def get_galileo_start_time():
        """
        Get the starting date and time of Galileo time.
        The Galileo System Time start epoch is defined as 13 seconds before midnight
        between 21st August and 22nd August 1999.
        
        Returns:
            datetime: Starting date and time of Galileo time (UTC).
        """
        return datetime(1999, 8, 22, 0, 0, 0) - timedelta(seconds=13)

    @staticmethod
    def get_galileo_time(input_datetime):
        """
        Get the Galileo week number and time of week (tow) in seconds from the input datetime.
        
        Parameters:
            input_datetime (datetime): The input date and time in UTC.
            
        Returns:
            tuple: (galileo_week, tow)
        """
        galileo_start = GalileoTime.get_galileo_start_time()
        seconds_per_week = 7 * 24 * 60 * 60  # Total seconds in a week

        # Calculate total seconds from the Galileo epoch to the input time
        total_seconds_since_epoch = (input_datetime - galileo_start).total_seconds()

        # Compute the current Galileo week number
        galileo_week = int(total_seconds_since_epoch // seconds_per_week)

        # Calculate the time of week (tow)
        tow = total_seconds_since_epoch % seconds_per_week

        return galileo_week, int(tow)

    @staticmethod
    def get_local_time(galileo_week, tow, tzone='UTC'):
        """
        Get local time as a datetime object from Galileo week and time of week.
        
        Parameters:
            galileo_week (int): Galileo week number.
            tow (int): Time of week in seconds.
            tzone (str): Time zone string (not used in this implementation).
            
        Returns:
            datetime: Local time corresponding to the given Galileo week and tow.
        """
        galileo_start = GalileoTime.get_galileo_start_time()
        # Compute the local time in UTC
        local_time = galileo_start + timedelta(weeks=galileo_week, seconds=tow)
        return local_time



import numpy as np

class GNSSConstellationGalAlmanac:
    def __init__(self):
        # Names of absolutely required parameters
        self.RequiredVariableNames = [
            'SVID', 'aSqRoot', 'ecc', 'deltai', 'omega0',
            'omegaDot', 'w', 'm0', 't0a', 'wna'
        ]

    @staticmethod
    def get_gnss_time(input_datetime):
        """
        Get GNSS week number and time of week (tow) in seconds from user input datetime.
        
        Parameters:
            input_datetime (datetime): The input date and time in UTC.
        
        Returns:
            tuple: (galileo_week, tow)
        """
        # Use the GalileoTime class to compute the Galileo week and ToW
        galileo_week, tow = GalileoTime.get_galileo_time(input_datetime)
        return galileo_week, tow

    @staticmethod
    def extract_orbit_parameters(orbit_params):
        """
        Extract orbit parameters from input almanac dictionary.
        """
        # Number of satellites for which parameters are available
        num_satellites = len(orbit_params['SVID'])
        
        # Factor to convert parameters in Semicircles to Radians
        semicir2rad_factor = np.pi

        # Extract the orbital parameters
        week_num = orbit_params['wna']
        toe = orbit_params['t0a']
        a_ref = (orbit_params['aSqRoot'] + np.sqrt(29600000))**2
        m0 = orbit_params['m0'] * semicir2rad_factor
        e0 = orbit_params['ecc']
        omega0 = orbit_params['w'] * semicir2rad_factor
        
        # i0 is set to 56 degrees in radians for all satellites
        i0 = np.deg2rad(56) * np.ones(num_satellites)
        i_del = orbit_params['deltai'] * semicir2rad_factor
        omega_ref_dot = orbit_params['omegaDot'] * semicir2rad_factor
        omega0 = orbit_params['omega0'] * semicir2rad_factor
        sat_ids = orbit_params['SVID']

        # Set the remaining orbital parameters to zero
        zero_vector = np.zeros(num_satellites)
        delta_a = zero_vector
        a_dot = zero_vector
        delta_n0 = zero_vector
        delta_n0_dot = zero_vector
        i_dot = zero_vector
        cis = zero_vector
        cic = zero_vector
        crs = zero_vector
        crc = zero_vector
        cus = zero_vector
        cuc = zero_vector
        delta_omega_dot = zero_vector

        return (week_num, toe, a_ref, delta_a, a_dot, delta_n0, delta_n0_dot,
                m0, e0, omega0, i0, i_dot, i_del, cis, cic, crs, crc,
                cus, cuc, omega_ref_dot, omega0, delta_omega_dot, sat_ids)

    def compute_position_velocity(self, almanac, t):
        """
        Compute position and velocity of satellites using almanac data.
        """
        # Get GNSS time (week number and time of week)
        gnss_week, tow = self.get_gnss_time(t)

        # Extract orbit parameters
        orbit_params = {
            'SVID': np.array([1]),  # Example Satellite ID
            'aSqRoot': np.array([almanac['aSqRoot']]),
            'ecc': np.array([almanac['ecc']]),
            'deltai': np.array([almanac['deltai']]),
            'omega0': np.array([almanac['omega0']]),
            'omegaDot': np.array([almanac['omegaDot']]),
            'w': np.array([almanac['w']]),
            'm0': np.array([almanac['m0']]),
            't0a': np.array([almanac['t0a']]),
            'wna': np.array([0])  # Set to 0 for example
        }
        (week_num, toe, a_ref, delta_a, a_dot, delta_n0, delta_n0_dot,
         m0, e0, omega0, i0, i_dot, i_del, cis, cic, crs, crc,
         cus, cuc, omega_ref_dot, omega0, delta_omega_dot, sat_ids) = self.extract_orbit_parameters(orbit_params)

        # Constants for the calculation
        mu = 3.986005e14  # Earth's gravitational constant, in m^3/s^2
        omega_e_dot = 7.2921151467e-5  # Earth's rotation rate, in rad/s

        # Call the orbit_parameters_to_ecef function to compute position and velocity
        pos, vel = orbit_parameters_to_ecef(
            gps_week=gnss_week,
            t=tow,
            week_num=week_num[0],
            toe=toe[0],
            ARef=a_ref[0],
            deltaA=delta_a[0],
            ADot=a_dot[0],
            mu=mu,
            deltan0=delta_n0[0],
            deltan0Dot=delta_n0_dot[0],
            M0=m0[0],
            e0=e0[0],
            omega0=omega0[0],
            i0=i0[0],
            iDot=i_dot[0],
            Cis=cis[0],
            Cic=cic[0],
            Crs=crs[0],
            Crc=crc[0],
            Cus=cus[0],
            Cuc=cuc[0],
            OmegaEDot=omega_e_dot,
            OmegaRefDot=omega_ref_dot[0],
            Omega0=omega0[0],
            deltaOmegaDot=delta_omega_dot[0]
        )

        return pos, vel

# Example usage:
# almanac_data = {
#     "aSqRoot": 0.03515625,  # Square root of semi-major axis
#     "ecc": 0.00030517578125,  # Eccentricity
#     "deltai": -0.00372314453125063230670699354619,  # Inclination change
#     "omega0": -0.797332763671874888977697537484,  # Right ascension
#     "omegaDot": -1.86264514923107200935514487085e-09,  # Rate of right ascension
#     "w": 0.0497131347656228281262080770375,  # Argument of perigee
#     "m0": 0.0402221679687564670491184415368,  # Mean anomaly at reference time
#     "af0": 0.0001678466796875,  # Clock correction bias
#     "af1": 3.63797880709199971837642219221e-12,  # Clock correction rate
#     "t0a": 204000,  # Almanac reference time in seconds
# }





# from datetime import datetime, timedelta

# def seconds_from_week_start(input_datetime):
#     """
#     Calculate the number of seconds from the start of the current Galileo week.
    
#     Parameters:
#         input_datetime (datetime): The input date and time in UTC.
        
#     Returns:
#         int: Number of seconds from the start of the current Galileo week.
#         int: Current Galileo week number.
#     """
#     # Galileo epoch: August 22, 1999, 00:00:00 UTC
#     galileo_epoch = datetime(1999, 8, 22, 0, 0, 0)
#     seconds_per_week = 7 * 24 * 60 * 60  # Total seconds in a week

#     # Calculate total seconds from the Galileo epoch to the input time
#     total_seconds_since_epoch = (input_datetime - galileo_epoch).total_seconds()

#     # Compute the current Galileo week number
#     current_week = int(total_seconds_since_epoch // seconds_per_week)

#     # Find the start time of the current Galileo week
#     start_of_current_week = galileo_epoch + timedelta(weeks=current_week)

#     # Calculate the number of seconds from the start of the current week
#     seconds_from_start_of_week = (input_datetime - start_of_current_week).total_seconds()

#     return int(seconds_from_start_of_week), current_week

# # Example usage:
# input_datetime = datetime(2024, 10, 16, 12, 0, 0)  # Example date: October 16, 2024, 12:00:00 UTC
# t, current_week = seconds_from_week_start(input_datetime)
# print("Seconds from the start of the current Galileo week:", t)
# print("Current Galileo week number:", current_week)




# # Example usage:
# # input_datetime = datetime(2024, 10, 16, 15, 50, 0)  # Example date: October 16, 2024, 12:00:00 UTC
# input_datetime = datetime.now()
# galileo_week, tow = GalileoTime.get_galileo_time(input_datetime)
# print("Galileo Week Number:", galileo_week)
# print("Time of Week (in seconds):", tow)

# # Get local time from Galileo week and ToW
# local_time = GalileoTime.get_local_time(galileo_week, tow)
# print("Local Time from Galileo Week and ToW:", local_time)



# # t = 256000  # Example current time in seconds

# gnss_constellation = GNSSConstellationGalAlmanac()
# position, velocity = gnss_constellation.compute_position_velocity(almanac_data, input_datetime)

# print("Position:", position)
# print("Velocity:", velocity)




# # Create an instance of the almanac class
# gnss_almanac = GNSSConstellationGalAlmanac()

# # Sample almanac data
# almanac_data = {
#     "aSqRoot": 0.03515625,  # Square root of semi-major axis
#     "ecc": 0.00030517578125,  # Eccentricity
#     "deltai": -0.00372314453125063230670699354619,  # Inclination change
#     "omega0": -0.797332763671874888977697537484,  # Right ascension
#     "omegaDot": -1.86264514923107200935514487085e-09,  # Rate of right ascension
#     "w": 0.0497131347656228281262080770375,  # Argument of perigee
#     "m0": 0.0402221679687564670491184415368,  # Mean anomaly at reference time
#     "af0": 0.0001678466796875,  # Clock correction bias
#     "af1": 3.63797880709199971837642219221e-12,  # Clock correction rate
#     "t0a": 204000,  # Almanac reference time in seconds
# }

# # Date for computation
# date_for_computation = datetime(2024, 10, 16, 12, 0, 0)

# # Compute position and velocity
# position, velocity = gnss_almanac.compute_position_velocity(almanac_data, date_for_computation)

# print("Satellite Position (ECEF):", position)
# print("Satellite Velocity (ECEF):", velocity)









'''
GPS COORDS
'''

# import numpy as np
# from datetime import datetime, timedelta
# from math import radians, sin, cos, sqrt, atan2, pi

# # Constants
# mu = 3.986004418e14  # Earth's gravitational parameter, m^3/s^2
# R_earth = 6378137  # Earth's radius, m
# J2 = 1.082629989e-3  # J2 gravitational perturbation coefficient

# # Satellite orbital parameters
# satellite_params =  {
#     "OBJECT_NAME": "GPS BIIRM-4 (PRN 15)",
#     "OBJECT_ID": "2007-047A",
#     "EPOCH": "2024-10-13T23:09:23.051232",
#     "MEAN_MOTION": 2.00558427,
#     "ECCENTRICITY": 0.0159318,
#     "INCLINATION": 53.6732,
#     "RA_OF_ASC_NODE": 105.3281,
#     "ARG_OF_PERICENTER": 78.0662,
#     "MEAN_ANOMALY": 283.7736,
#     "EPHEMERIS_TYPE": 0,
#     "CLASSIFICATION_TYPE": "U",
#     "NORAD_CAT_ID": 32260,
#     "ELEMENT_SET_NO": 999,
#     "REV_AT_EPOCH": 12455,
#     "BSTAR": 0,
#     "MEAN_MOTION_DOT": 7.6e-7,
#     "MEAN_MOTION_DDOT": 0
#   }

# # Convert epoch to datetime
# epoch = datetime.fromisoformat(satellite_params["EPOCH"])

# # Calculate the time elapsed since the epoch in seconds
# current_time = datetime.utcnow()  # Use current UTC time
# time_elapsed = (current_time - epoch).total_seconds()

# # Mean motion in rad/s
# mean_motion = satellite_params["MEAN_MOTION"] * 2 * pi / 86400  # rev/day to rad/s
# mean_anomaly = radians(satellite_params["MEAN_ANOMALY"]) + mean_motion * time_elapsed

# # Eccentric anomaly using Kepler's equation
# eccentricity = satellite_params["ECCENTRICITY"]
# E = mean_anomaly  # Initial guess
# for _ in range(10):  # Iterate to solve for eccentric anomaly
#     E = mean_anomaly + eccentricity * sin(E)

# # Calculate true anomaly
# true_anomaly = 2 * atan2(sqrt(1 + eccentricity) * sin(E / 2), sqrt(1 - eccentricity) * cos(E / 2))

# # Semi-major axis in meters
# n = satellite_params["MEAN_MOTION"] * 2 * pi / 86400  # rev/day to rad/s
# a = (mu / (n ** 2)) ** (1 / 3)

# # Orbital radius
# r = a * (1 - eccentricity * cos(E))

# # Calculate orbital position in ECI
# inclination = radians(satellite_params["INCLINATION"])
# ra_of_asc_node = radians(satellite_params["RA_OF_ASC_NODE"])
# arg_of_pericenter = radians(satellite_params["ARG_OF_PERICENTER"])

# x_orb = r * (cos(arg_of_pericenter) * cos(true_anomaly) - sin(arg_of_pericenter) * sin(true_anomaly) * cos(inclination))
# y_orb = r * (sin(arg_of_pericenter) * cos(true_anomaly) + cos(arg_of_pericenter) * sin(true_anomaly) * cos(inclination))
# z_orb = r * (sin(true_anomaly) * sin(inclination))

# # Transform ECI to ECEF coordinates
# # Earth rotation rate (rad/s)
# omega_e = 7.2921159e-5

# # Time since epoch in seconds
# t = time_elapsed

# # Rotate ECI to ECEF
# theta = omega_e * t  # Earth rotation angle

# # ECEF coordinates
# x_ecef = x_orb * cos(theta) + y_orb * sin(theta)
# y_ecef = -x_orb * sin(theta) + y_orb * cos(theta)
# z_ecef = z_orb

# # ECEF coordinates result
# print(x_ecef, y_ecef, z_ecef)






'''
GALILEO
'''

# # Satellite orbital parameters for GSAT0201 (GALILEO 5)
# galileo_params = {
#     "OBJECT_NAME": "GSAT0201 (GALILEO 5)",
#     "OBJECT_ID": "2014-050A",
#     "EPOCH": "2024-10-15T06:46:42.696768",
#     "MEAN_MOTION": 1.85519395,
#     "ECCENTRICITY": 0.1614137,
#     "INCLINATION": 49.5361,
#     "RA_OF_ASC_NODE": 298.2287,
#     "ARG_OF_PERICENTER": 155.9444,
#     "MEAN_ANOMALY": 212.5362,
#     "REV_AT_EPOCH": 6700,
#     "MEAN_MOTION_DOT": -3.2e-7,
# }

# # Convert epoch to datetime
# galileo_epoch = datetime.fromisoformat(galileo_params["EPOCH"])

# # Calculate the time elapsed since the epoch in seconds
# galileo_time_elapsed = (current_time - galileo_epoch).total_seconds()

# # Mean motion in rad/s
# galileo_mean_motion = galileo_params["MEAN_MOTION"] * 2 * pi / 86400  # rev/day to rad/s
# galileo_mean_anomaly = radians(galileo_params["MEAN_ANOMALY"]) + galileo_mean_motion * galileo_time_elapsed

# # Eccentric anomaly using Kepler's equation
# galileo_eccentricity = galileo_params["ECCENTRICITY"]
# E_galileo = galileo_mean_anomaly  # Initial guess
# for _ in range(10):  # Iterate to solve for eccentric anomaly
#     E_galileo = galileo_mean_anomaly + galileo_eccentricity * sin(E_galileo)

# # Calculate true anomaly
# galileo_true_anomaly = 2 * atan2(sqrt(1 + galileo_eccentricity) * sin(E_galileo / 2), sqrt(1 - galileo_eccentricity) * cos(E_galileo / 2))

# # Semi-major axis in meters
# galileo_n = galileo_params["MEAN_MOTION"] * 2 * pi / 86400  # rev/day to rad/s
# galileo_a = (mu / (galileo_n ** 2)) ** (1 / 3)

# # Orbital radius
# galileo_r = galileo_a * (1 - galileo_eccentricity * cos(E_galileo))

# # Calculate orbital position in ECI
# galileo_inclination = radians(galileo_params["INCLINATION"])
# galileo_ra_of_asc_node = radians(galileo_params["RA_OF_ASC_NODE"])
# galileo_arg_of_pericenter = radians(galileo_params["ARG_OF_PERICENTER"])

# galileo_x_orb = galileo_r * (cos(galileo_arg_of_pericenter) * cos(galileo_true_anomaly) - sin(galileo_arg_of_pericenter) * sin(galileo_true_anomaly) * cos(galileo_inclination))
# galileo_y_orb = galileo_r * (sin(galileo_arg_of_pericenter) * cos(galileo_true_anomaly) + cos(galileo_arg_of_pericenter) * sin(galileo_true_anomaly) * cos(galileo_inclination))
# galileo_z_orb = galileo_r * (sin(galileo_true_anomaly) * sin(galileo_inclination))

# # Transform ECI to ECEF coordinates
# galileo_theta = omega_e * galileo_time_elapsed  # Earth rotation angle

# # ECEF coordinates
# galileo_x_ecef = galileo_x_orb * cos(galileo_theta) + galileo_y_orb * sin(galileo_theta)
# galileo_y_ecef = -galileo_x_orb * sin(galileo_theta) + galileo_y_orb * cos(galileo_theta)
# galileo_z_ecef = galileo_z_orb

# # ECEF coordinates result for GSAT0201
# galileo_coordinates = (galileo_x_ecef, galileo_y_ecef, galileo_z_ecef)
# print(galileo_coordinates)










'''
	WORLD VIEW
'''





# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# from datetime import datetime, timedelta
# from math import radians, sin, cos, atan2, sqrt, pi, degrees

# # Function to calculate ECEF coordinates from orbital parameters
# def calculate_ecef(satellite_params, duration_hours):
#     # Constants
#     mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
#     omega_e = 7.2921159e-5  # Earth's rotation rate (rad/s)
    
#     # Satellite parameters
#     print('HEREREREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE', satellite_params)
#     epoch = datetime.fromisoformat(satellite_params["EPOCH"])
#     mean_motion = satellite_params["MEAN_MOTION"] * 2 * pi / 86400  # rev/day to rad/s
#     eccentricity = satellite_params["ECCENTRICITY"]
#     inclination = radians(satellite_params["INCLINATION"])
#     ra_of_asc_node = radians(satellite_params["RA_OF_ASC_NODE"])
#     arg_of_pericenter = radians(satellite_params["ARG_OF_PERICENTER"])
    
#     # Time intervals in seconds
#     time_steps = np.arange(0, duration_hours * 3600, 60)  # 1-minute intervals
#     positions = []
    
#     for t in time_steps:
#         # Calculate the mean anomaly at time t
#         mean_anomaly = radians(satellite_params["MEAN_ANOMALY"]) + mean_motion * t
#         E = mean_anomaly  # Initial guess for eccentric anomaly
        
#         # Solve Kepler's equation for eccentric anomaly
#         for _ in range(10):
#             E = mean_anomaly + eccentricity * sin(E)
        
#         # Calculate true anomaly
#         true_anomaly = 2 * atan2(sqrt(1 + eccentricity) * sin(E / 2), sqrt(1 - eccentricity) * cos(E / 2))
        
#         # Calculate orbital radius
#         a = (mu / (mean_motion ** 2)) ** (1 / 3)  # semi-major axis
#         r = a * (1 - eccentricity * cos(E))
        
#         # Calculate position in ECI
#         x_orb = r * (cos(arg_of_pericenter) * cos(true_anomaly) - sin(arg_of_pericenter) * sin(true_anomaly) * cos(inclination))
#         y_orb = r * (sin(arg_of_pericenter) * cos(true_anomaly) + cos(arg_of_pericenter) * sin(true_anomaly) * cos(inclination))
#         z_orb = r * (sin(true_anomaly) * sin(inclination))
        
#         # Transform ECI to ECEF
#         theta = omega_e * t  # Earth rotation angle
#         x_ecef = x_orb * cos(theta) + y_orb * sin(theta)
#         y_ecef = -x_orb * sin(theta) + y_orb * cos(theta)
#         z_ecef = z_orb
        
#         positions.append((x_ecef, y_ecef, z_ecef))
    
#     return np.array(positions)

# # Function to convert ECEF to latitude and longitude
# def ecef_to_lat_lon(ecef_coords):
#     x, y, z = ecef_coords
#     r = sqrt(x**2 + y**2)
#     lat = atan2(z, r) * (180 / pi)  # Latitude in degrees
#     lon = atan2(y, x) * (180 / pi)  # Longitude in degrees
#     return lat, lon

# # Function to plot satellite paths on a map
# def plot_satellite_paths(sat_params_list, duration_hours):
#     plt.figure(figsize=(12, 6))
#     ax = plt.axes(projection=ccrs.PlateCarree())  # Using Plate Carree projection
#     ax.coastlines()
#     ax.set_title(f"Satellite Paths over {duration_hours} hours")
    
#     for sat_params in sat_params_list:
#         ecef_positions = calculate_ecef(sat_params, duration_hours)
#         lat_lon_positions = np.array([ecef_to_lat_lon(pos) for pos in ecef_positions])
#         ax.plot(lat_lon_positions[:, 1], lat_lon_positions[:, 0], marker='o', markersize=2, linewidth=1, label=sat_params['OBJECT_NAME'])
    
#     plt.legend()
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.grid()
#     plt.show()



# # Plot the paths for the given satellites over 24 hours
# # duration_hours = 24
# # plot_satellite_paths(satellite_params, duration_hours)




# '''
# 	DOP
# '''

import numpy as np
from math import radians, degrees, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt

# WGS84 ellipsoid constants
a = 6378137.0        # Semi-major axis in meters
f = 1 / 298.257223563  # Flattening
e2 = f * (2 - f)     # Square of eccentricity

# Function to convert geodetic coordinates (lat, lon, alt) to ECEF coordinates
def geodetic_to_ecef(lat, lon, alt):
    lat, lon = radians(lat), radians(lon)
    
    # Calculate prime vertical radius of curvature
    N = a / sqrt(1 - e2 * sin(lat) ** 2)
    
    # Calculate ECEF coordinates
    x = (N + alt) * cos(lat) * cos(lon)
    y = (N + alt) * cos(lat) * sin(lon)
    z = (N * (1 - e2) + alt) * sin(lat)
    
    return x, y, z

# Function to convert ECEF to ENU coordinates for DOP calculation
def ecef_to_enu(receiver_position, satellite_position):
    lat, lon, alt = receiver_position
    lat, lon = radians(lat), radians(lon)
    
    # Rotation matrix from ECEF to ENU
    R = np.array([
        [-sin(lon), cos(lon), 0],
        [-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat)],
        [cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)]
    ])
    
    # ECEF coordinates of the receiver
    x0, y0, z0 = geodetic_to_ecef(lat, lon, alt)
    dx, dy, dz = satellite_position - np.array([x0, y0, z0])
    
    # Transform to ENU
    enu = np.dot(R, np.array([dx, dy, dz]))
    return enu

# Function to calculate Geometric DOP (GDOP), Position DOP (PDOP), and Horizontal DOP (HDOP)
def calculate_dop(receiver_position, visible_satellites):
    H = []
    
    for sat_pos in visible_satellites:
        enu = ecef_to_enu(receiver_position, sat_pos)
        d = np.linalg.norm(enu)
        H.append([enu[0] / d, enu[1] / d, enu[2] / d, 1.0])
    
    H = np.array(H)
    
    # Check if there are enough satellites for the calculation (at least 4)
    if H.shape[0] < 4:
        return None, None, None  # Indicate insufficient satellite coverage
    
    Q = np.linalg.inv(np.dot(H.T, H))
    
    GDOP = np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2] + Q[3, 3])
    PDOP = np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])
    HDOP = np.sqrt(Q[0, 0] + Q[1, 1])
    
    return GDOP, PDOP, HDOP

# Function to compute satellite visibility based on elevation
def compute_visibility(receiver_position, satellite_positions, elevation_mask=10):
    visible_satellites = []
    
    for sat_pos in satellite_positions:
        enu = ecef_to_enu(receiver_position, sat_pos)
        elevation = degrees(atan2(enu[2], sqrt(enu[0]**2 + enu[1]**2)))
        if elevation >= elevation_mask:
            visible_satellites.append(sat_pos)
    
    return visible_satellites

# # Function to plot the DOP values over time
# def plot_dop_over_time(dop_values, duration_hours):
#     times = np.linspace(0, duration_hours, len(dop_values))
#     GDOP_values, PDOP_values, HDOP_values = zip(*[(gdop, pdop, hdop) if gdop is not None else (np.nan, np.nan, np.nan) for gdop, pdop, hdop in dop_values])
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(times, GDOP_values, label='GDOP', color='r')
#     plt.plot(times, PDOP_values, label='PDOP', color='b')
#     plt.plot(times, HDOP_values, label='HDOP', color='g')
    
#     plt.title('DOP Values Over Time')
#     plt.xlabel('Time (hours)')
#     plt.ylabel('DOP')
#     plt.legend()
#     plt.grid()
#     plt.show()

# # Receiver position: latitude, longitude, altitude in meters
# receiver_position = (45.0703, 7.6869, 300)  # Example location (Turin, Italy)

# # Example: Compute positions and visibility for the given satellites over 6 hours
# duration_hours = 24
# ecef_positions_list = [calculate_ecef(sat_params, duration_hours) for sat_params in satellite_params]

# # Determine visibility for the receiver
# all_visible_satellites = [compute_visibility(receiver_position, ecef_positions) for ecef_positions in ecef_positions_list]

# # Calculate DOP for each time step
# dop_values = [calculate_dop(receiver_position, visible_satellites) for visible_satellites in all_visible_satellites]

# # Plot the DOP values over time
# plot_dop_over_time(dop_values, duration_hours)

# # Output DOP values for inspection
# for i, (GDOP, PDOP, HDOP) in enumerate(dop_values):
#     if GDOP is not None:
#         print(f"Time step {i}: GDOP = {GDOP:.2f}, PDOP = {PDOP:.2f}, HDOP = {HDOP:.2f}")
#     else:
#         print(f"Time step {i}: Insufficient satellite coverage for DOP calculation.")








import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from math import radians, sin, cos, atan2, sqrt, pi, degrees

# Updated function to calculate ECEF positions based on time
def calculate_ecef(satellite_params, current_time):
    # Constants
    mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
    omega_e = 7.2921159e-5  # Earth's rotation rate (rad/s)

    # Satellite parameters
    epoch = datetime.fromisoformat(satellite_params["EPOCH"])
    mean_motion = satellite_params["MEAN_MOTION"] * 2 * pi / 86400  # rev/day to rad/s
    eccentricity = satellite_params["ECCENTRICITY"]
    inclination = radians(satellite_params["INCLINATION"])
    ra_of_asc_node = radians(satellite_params["RA_OF_ASC_NODE"])
    arg_of_pericenter = radians(satellite_params["ARG_OF_PERICENTER"])

    # Time since epoch in seconds
    delta_t = (current_time - epoch).total_seconds()

    # Calculate the mean anomaly at time delta_t
    mean_anomaly = radians(satellite_params["MEAN_ANOMALY"]) + mean_motion * delta_t
    E = mean_anomaly  # Initial guess for eccentric anomaly

    # Solve Kepler's equation for eccentric anomaly
    for _ in range(10):
        E = mean_anomaly + eccentricity * sin(E)

    # Calculate true anomaly
    true_anomaly = 2 * atan2(sqrt(1 + eccentricity) * sin(E / 2), sqrt(1 - eccentricity) * cos(E / 2))

    # Calculate orbital radius
    a = (mu / (mean_motion ** 2)) ** (1 / 3)  # semi-major axis
    r = a * (1 - eccentricity * cos(E))

    # Calculate position in ECI
    x_orb = r * (cos(arg_of_pericenter) * cos(true_anomaly) - sin(arg_of_pericenter) * sin(true_anomaly) * cos(inclination))
    y_orb = r * (sin(arg_of_pericenter) * cos(true_anomaly) + cos(arg_of_pericenter) * sin(true_anomaly) * cos(inclination))
    z_orb = r * (sin(true_anomaly) * sin(inclination))

    # Transform ECI to ECEF
    theta = omega_e * delta_t  # Earth rotation angle
    x_ecef = x_orb * cos(theta) + y_orb * sin(theta)
    y_ecef = -x_orb * sin(theta) + y_orb * cos(theta)
    z_ecef = z_orb

    return np.array([x_ecef, y_ecef, z_ecef])

# Updated function to compute DOP values for a list of satellites over time
def compute_dop_over_time(sat_params_list, receiver_position, start_time, duration_hours, resolution_seconds=900):
    # Generate the time steps based on the start time, duration, and resolution
    time_steps = [start_time + timedelta(seconds=i) for i in range(0, duration_hours * 3600, resolution_seconds)]
    dop_data = {"dop": []}

    # Loop over each time step to calculate satellite positions and DOP
    for current_time in time_steps:
        # Calculate the ECEF positions for all satellites at the current time
        ecef_positions = [calculate_ecef(sat_params, current_time) for sat_params in sat_params_list]

        # Determine the satellites visible from the receiver at the current time
        visible_satellites = compute_visibility(receiver_position, ecef_positions)

        # Compute DOP values based on visible satellites
        dop = calculate_dop(receiver_position, visible_satellites)

        # Store DOP values in the dictionary
        if dop[0] is not None:  # Check if DOP values are valid
            dop_data["dop"].append({
                current_time.strftime('%H:%M'): {
                    "GDOP": float(dop[0]),
                    "PDOP": float(dop[1]),
                    "HDOP": float(dop[2])
                }
            })
        else:
            dop_data["dop"].append({
                current_time.strftime('%H:%M'): {
                    "GDOP": None,
                    "PDOP": None,
                    "HDOP": None
                }
            })

    return dop_data


# Function to plot the DOP values over time with actual datetime values on the x-axis
def plot_dop_over_time_with_datetime(time_steps, dop_values):
    # Extract GDOP, PDOP, and HDOP values
    GDOP_values, PDOP_values, HDOP_values = zip(*[(gdop, pdop, hdop) if gdop is not None else (np.nan, np.nan, np.nan) for gdop, pdop, hdop in dop_values])

    # Plot DOP values over time
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, GDOP_values, label='GDOP', color='r')
    plt.plot(time_steps, PDOP_values, label='PDOP', color='b')
    plt.plot(time_steps, HDOP_values, label='HDOP', color='g')

    plt.title('DOP Values Over Time')
    plt.xlabel('Time')
    plt.ylabel('DOP')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# # Example usage
# # Receiver position: latitude, longitude, altitude in meters
# receiver_position = (45.0703, 7.6869, 300)  # Example location (Turin, Italy)

# # Start time and duration for the analysis
# start_time = datetime(2024, 10, 17, 12, 0, 0)  # Example start time
# duration_hours = 6  # Duration in hours

# # Calculate DOP over time and plot the results
# time_steps, dop_values = compute_dop_over_time(satellite_params, receiver_position, start_time, duration_hours)
# plot_dop_over_time_with_datetime(time_steps, dop_values)

# # Output DOP values for inspection
# for time, (GDOP, PDOP, HDOP) in zip(time_steps, dop_values):
#     if GDOP is not None:
#         print(f"{time}: GDOP = {GDOP:.2f}, PDOP = {PDOP:.2f}, HDOP = {HDOP:.2f}")
#     else:
#         print(f"{time}: Insufficient satellite coverage for DOP calculation.")


# Example usage
# Receiver position: latitude, longitude, altitude in meters
receiver_position = (45.0703, 7.6869, 300)  # Example location (Turin, Italy)

# Start time and duration for the analysis
start_time = datetime(2024, 10, 17, 12, 0, 0)  # Example start time
duration_hours = 6  # Duration in hours

# Calculate DOP over time
dop_data = compute_dop_over_time(satellite_params, receiver_position, start_time, duration_hours)

# Output DOP values for inspection
print(dop_data)



'''
	SKYPLOT, VISBILITY
'''




# import matplotlib.pyplot as plt
# import numpy as np
# from math import radians, degrees, sin, cos, atan2, sqrt

# Function to compute azimuth and elevation from ENU coordinates
def enu_to_az_el(enu):
    e, n, u = enu
    horizontal_distance = sqrt(e**2 + n**2)
    elevation = degrees(atan2(u, horizontal_distance))
    azimuth = degrees(atan2(e, n)) % 360
    return azimuth, elevation

# # Function to compute satellite visibility over time
# def compute_visibility_over_time(receiver_position, satellite_positions_list, elevation_mask=10):
#     visibility_over_time = []
    
#     for satellite_positions in satellite_positions_list:
#         visible_satellites_at_time = []
#         for i, sat_pos in enumerate(satellite_positions):
#             enu = ecef_to_enu(receiver_position, sat_pos)
#             _, elevation = enu_to_az_el(enu)
#             if elevation >= elevation_mask:
#                 visible_satellites_at_time.append(i)  # Using index to identify satellites
#         visibility_over_time.append(visible_satellites_at_time)
    
#     return visibility_over_time

# # Function to plot visibility over time
# def plot_visibility_over_time(visibility_over_time, duration_hours, satellite_params_list):
#     times = np.linspace(0, duration_hours, len(visibility_over_time))
#     plt.figure(figsize=(10, 6))
    
#     for i, sat_params in enumerate(satellite_params_list):
#         visibility = [i in visible for visible in visibility_over_time]
#         plt.plot(times, visibility, label=sat_params["OBJECT_NAME"], linewidth=1.5)
    
#     plt.title("Satellite Visibility Over Time")
#     plt.xlabel("Time (hours)")
#     plt.ylabel("Visibility (1=Visible, 0=Not Visible)")
#     plt.yticks([0, 1])
#     plt.legend(loc='upper right', fontsize='small')
#     plt.grid()
#     plt.show()

# # Function to create a skyplot for visible satellites at the receiver
# def skyplot(receiver_position, satellite_positions_list, elevation_mask=10):
#     az_el_data = []

#     for satellite_positions in satellite_positions_list:
#         for sat_pos in satellite_positions:
#             enu = ecef_to_enu(receiver_position, sat_pos)
#             azimuth, elevation = enu_to_az_el(enu)
#             if elevation >= elevation_mask:
#                 az_el_data.append((azimuth, elevation))

#     # Skyplot
#     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
#     ax.set_theta_zero_location('N')  # Zero degrees at the top (North)
#     ax.set_theta_direction(-1)       # Plot clockwise
    
#     # Plot satellites
#     for azimuth, elevation in az_el_data:
#         ax.scatter(radians(azimuth), 90 - elevation, c='b', s=50, alpha=0.75)

#     ax.set_title("Skyplot of Visible Satellites")
#     ax.set_ylim(0, 90)
#     ax.set_yticks(range(0, 91, 15))
#     ax.set_yticklabels([f'{90 - el}°' for el in range(0, 91, 15)])
#     ax.set_xticks(np.radians(np.arange(0, 360, 45)))
#     ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

#     plt.show()

# # Example usage
# receiver_position = (45.0703, 7.6869, 300)  # Example location (Turin, Italy)
# duration_hours = 3

# # List of ECEF positions for the given satellites
# ecef_positions_list = [calculate_ecef(sat_params, duration_hours) for sat_params in satellite_params_list]

# # Compute visibility over time for the receiver
# visibility_over_time = compute_visibility_over_time(receiver_position, ecef_positions_list)

# # Plot the visibility over time
# plot_visibility_over_time(visibility_over_time, duration_hours, satellite_params_list)

# # Generate the skyplot for the receiver position
# skyplot(receiver_position, ecef_positions_list)




# import matplotlib.pyplot as plt
# import numpy as np
# from math import radians, degrees, sin, cos, atan2, sqrt

# # Optimized function to compute satellite visibility over time
# def compute_cumulative_visibility(receiver_position, satellite_positions_list, elevation_mask=10):
#     visibility_count = []

#     for satellite_positions in satellite_positions_list:
#         count = 0
#         for sat_pos in satellite_positions:
#             enu = ecef_to_enu(receiver_position, sat_pos)
#             _, elevation = enu_to_az_el(enu)
#             if elevation >= elevation_mask:
#                 count += 1
#         visibility_count.append(count)

#     return visibility_count

# # Improved function to plot the number of visible satellites over time
# def plot_cumulative_visibility_over_time(visibility_count, duration_hours):
#     times = np.linspace(0, duration_hours, len(visibility_count))
#     plt.figure(figsize=(10, 6))

#     plt.plot(times, visibility_count, color='b', linewidth=1.5)
#     plt.title("Number of Visible Satellites Over Time")
#     plt.xlabel("Time (hours)")
#     plt.ylabel("Number of Visible Satellites")
#     plt.grid()
#     plt.show()

# # Optimized skyplot function
# def skyplot_optimized(receiver_position, satellite_positions_list, elevation_mask=10):
#     az_el_data = []

#     for satellite_positions in satellite_positions_list:
#         for sat_pos in satellite_positions:
#             enu = ecef_to_enu(receiver_position, sat_pos)
#             azimuth, elevation = enu_to_az_el(enu)
#             if elevation >= elevation_mask:
#                 az_el_data.append((azimuth, elevation))
#                 break  # Stop after finding one visible satellite per time step

#     # Skyplot with performance optimization
#     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
#     ax.set_theta_zero_location('N')  # Zero degrees at the top (North)
#     ax.set_theta_direction(-1)       # Plot clockwise
    
#     # Plot satellites
#     for azimuth, elevation in az_el_data:
#         ax.scatter(radians(azimuth), 90 - elevation, c='b', s=50, alpha=0.75)

#     ax.set_title("Skyplot of Visible Satellites")
#     ax.set_ylim(0, 90)
#     ax.set_yticks(range(0, 91, 15))
#     ax.set_yticklabels([f'{90 - el}°' for el in range(0, 91, 15)])
#     ax.set_xticks(np.radians(np.arange(0, 360, 45)))
#     ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

#     plt.show()

# # Example usage
# receiver_position = (45.0703, 7.6869, 300)  # Example location (Turin, Italy)
# duration_hours = 6

# # List of ECEF positions for the given satellites
# ecef_positions_list = [calculate_ecef(sat_params, duration_hours) for sat_params in satellite_params]

# # Generate the optimized skyplot for the receiver position
# skyplot_optimized(receiver_position, ecef_positions_list)




import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from math import radians, degrees, sin, cos, atan2, sqrt

# Function to convert ECEF coordinates to ENU coordinates
def ecef_to_enu(receiver_position, satellite_position):
    lat, lon, alt = receiver_position
    lat = radians(lat)
    lon = radians(lon)

    # Receiver ECEF position
    x_r = alt * cos(lat) * cos(lon)
    y_r = alt * cos(lat) * sin(lon)
    z_r = alt * sin(lat)

    # Satellite ECEF position
    x_s, y_s, z_s = satellite_position

    # Convert ECEF to ENU
    dx = x_s - x_r
    dy = y_s - y_r
    dz = z_s - z_r

    # ENU transformation matrix
    e = -sin(lon) * dx + cos(lon) * dy
    n = -sin(lat) * cos(lon) * dx - sin(lat) * sin(lon) * dy + cos(lat) * dz
    u = cos(lat) * cos(lon) * dx + cos(lat) * sin(lon) * dy + sin(lat) * dz

    return np.array([e, n, u])

# Function to convert ENU coordinates to azimuth and elevation
def enu_to_az_el(enu):
    e, n, u = enu
    azimuth = degrees(atan2(e, n)) % 360
    elevation = degrees(atan2(u, sqrt(e**2 + n**2)))
    return azimuth, elevation

# Function to compute satellite visibility over time and return azimuth-elevation pairs
def compute_az_el_over_time(receiver_position, sat_params_list, start_time, duration_hours, resolution_seconds=900, elevation_mask=10):
    # Generate the time steps based on the start time, duration, and resolution
    time_steps = [start_time + timedelta(seconds=i) for i in range(0, duration_hours * 3600, resolution_seconds)]
    az_el_data = {current_time: [] for current_time in time_steps}

    # Loop over each time step to calculate satellite positions and azimuth/elevation
    for current_time in time_steps:
        # Calculate the ECEF positions for all satellites at the current time
        ecef_positions = [calculate_ecef(sat_params, current_time) for sat_params in sat_params_list]

        # Loop through each satellite and calculate azimuth and elevation
        for sat_pos in ecef_positions:
            enu = ecef_to_enu(receiver_position, sat_pos)
            azimuth, elevation = enu_to_az_el(enu)
            if elevation >= elevation_mask:
                az_el_data[current_time].append((azimuth, elevation))

    return az_el_data

# Function to plot the number of visible satellites over time with actual datetime values on the x-axis
def plot_cumulative_visibility_over_time(az_el_data):
    visibility_count = [len(az_el_data[time]) for time in az_el_data]
    time_steps = list(az_el_data.keys())

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, visibility_count, color='b', linewidth=1.5)
    plt.title("Number of Visible Satellites Over Time")
    plt.xlabel("Time")
    plt.ylabel("Number of Visible Satellites")
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Optimized skyplot function to plot azimuth-elevation data for visible satellites
def skyplot_optimized(receiver_position, az_el_data, elevation_mask=10):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.set_theta_zero_location('N')  # Zero degrees at the top (North)
    # ax.set_theta_direction(-1)       # Plot clockwise

    # Plot satellites at each time step
    for time, az_el_list in az_el_data.items():
        for azimuth, elevation in az_el_list:
            ax.scatter(radians(azimuth), 90 - elevation, c='b', s=50, alpha=0.75)

    ax.set_title("Skyplot of Visible Satellites")
    ax.set_ylim(0, 90)
    ax.set_yticks(range(0, 91, 15))
    # ax.set_yticklabels([f'{90 - el}°' for el in range(0, 91, 15)])
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

    plt.show()

# Example usage
receiver_position = (45.0703, 7.6869, 300)  # Example location (Turin, Italy)
start_time = datetime(2024, 10, 17, 12, 0, 0)  # Example start time
duration_hours = 6


# Calculate azimuth and elevation over time and plot the results
az_el_data = compute_az_el_over_time(receiver_position, satellite_params, start_time, duration_hours)

plot_cumulative_visibility_over_time(az_el_data)
# skyplot_optimized(receiver_position, az_el_data)



# {
#     "dop": [(time, GDOP, PDOP, HDOP), (time, GDOP, PDOP, HDOP), ...],
#     "visibility": ,
#     'skyplot': {"sat_object": "", }
# }



import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from math import radians, degrees, sin, cos, atan2, sqrt

# Function to convert ECEF coordinates to ENU coordinates
def ecef_to_enu(receiver_position, satellite_position):
    lat, lon, alt = receiver_position
    lat = radians(lat)
    lon = radians(lon)

    # Receiver ECEF position
    x_r = alt * cos(lat) * cos(lon)
    y_r = alt * cos(lat) * sin(lon)
    z_r = alt * sin(lat)

    # Satellite ECEF position
    x_s, y_s, z_s = satellite_position

    # Convert ECEF to ENU
    dx = x_s - x_r
    dy = y_s - y_r
    dz = z_s - z_r

    # ENU transformation matrix
    e = -sin(lon) * dx + cos(lon) * dy
    n = -sin(lat) * cos(lon) * dx - sin(lat) * sin(lon) * dy + cos(lat) * dz
    u = cos(lat) * cos(lon) * dx + cos(lat) * sin(lon) * dy + sin(lat) * dz

    return np.array([e, n, u])

# Function to convert ENU coordinates to azimuth and elevation
def enu_to_az_el(enu):
    e, n, u = enu
    azimuth = degrees(atan2(e, n)) % 360
    elevation = degrees(atan2(u, sqrt(e**2 + n**2)))
    return azimuth, elevation


# Function to compute azimuth, elevation, and time data for each satellite
def compute_satellite_data_over_time(receiver_position, sat_params_list, start_time, duration_hours, resolution_seconds=900, elevation_mask=10):
    # Generate the time steps based on the start time, duration, and resolution
    time_steps = [start_time + timedelta(seconds=i) for i in range(0, duration_hours * 3600, resolution_seconds)]
    
    # Initialize the dictionary to store satellite data, using OBJECT_NAME as the key
    satellite_data = {
        sat_params['OBJECT_NAME']: {"azimuth": [], "elevation": [], "time": []}
        for sat_params in sat_params_list
    }

    # Loop over each time step to calculate satellite positions and azimuth/elevation
    for current_time in time_steps:
        # Calculate the ECEF positions for all satellites at the current time
        ecef_positions = [calculate_ecef(sat_params, current_time) for sat_params in sat_params_list]

        # Loop through each satellite and calculate azimuth and elevation
        for idx, (sat_pos, sat_params) in enumerate(zip(ecef_positions, sat_params_list)):
            enu = ecef_to_enu(receiver_position, sat_pos)
            azimuth, elevation = enu_to_az_el(enu)
            if elevation >= elevation_mask:
                # Use OBJECT_NAME as the key for the satellite
                sat_name = sat_params['OBJECT_NAME']
                satellite_data[sat_name]["azimuth"].append(azimuth)
                satellite_data[sat_name]["elevation"].append(elevation)
                satellite_data[sat_name]["time"].append(current_time.strftime('%H:%M'))

    return satellite_data


# Function to plot the optimized skyplot with organized satellite data
def skyplot_with_satellite_data(satellite_data):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.set_theta_zero_location('N')  # Zero degrees at the top (North)
    ax.set_theta_direction(-1)       # Plot clockwise

    # Plot data for each satellite
    for sat, data in satellite_data.items():
        azimuths = data["azimuth"]
        elevations = data["elevation"]
        ax.scatter([radians(az) for az in azimuths], [90 - el for el in elevations], label=sat, s=50, alpha=0.75)

    ax.set_title("Skyplot of Visible Satellites")
    ax.set_ylim(0, 90)
    ax.set_yticks(range(0, 91, 15))
    ax.set_yticklabels([f'{90 - el}°' for el in range(0, 91, 15)])
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.legend(loc='upper right')

    plt.show()

# Example usage
receiver_position = (45.0703, 7.6869, 300)  # Example location (Turin, Italy)
start_time = datetime(2024, 10, 17, 12, 0, 0)  # Example start time
duration_hours = 6

# Calculate satellite data (azimuth, elevation, time) over time
satellite_data = compute_satellite_data_over_time(receiver_position, satellite_params, start_time, duration_hours)

# Plot the skyplot with the organized data
skyplot_with_satellite_data(satellite_data)

# Print the organized data structure
print(satellite_data)
