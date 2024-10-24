import json
import cartopy.crs as ccrs
import cartopy
import matplotlib.pyplot as plt






# class Satellite:

# 	def __init__(self, tle):
# 		self.conf_path = "services/config.json"
# 		self.conf = json.load(open(self.conf_path))
# 		self.tle = tle
# 		self.const = self.__detect_const()

# 	def __detect_const(self):
# 		obj_name = self.tle["OBJECT_NAME"]
# 		for const, data in self.conf["constellations"].items():
# 			for sat in data["sats"]:
# 				if sat["OBJECT_NAME"] == obj_name:
# 					return const




import math
import json
from datetime import datetime, timedelta

class Satellite:

    def __init__(self, tle):
        self.conf_path = "config.json"
        self.conf = json.load(open(self.conf_path))
        self.tle = tle
        self.const = self.__detect_const()  # Call to detect the constellation (GPS, GAL, GLO, BEI)
        self.mu = 398600.4418  # Gravitational parameter for Earth [km^3/s^2]
        self.epoch = self.__parse_epoch()
        self.__apply_constellation_specific_settings()  # Adjust for constellation-specific properties

    def __detect_const(self):
        obj_name = self.tle["OBJECT_NAME"]
        for const, data in self.conf["constellations"].items():
            for sat in data["sats"]:
                if sat["OBJECT_NAME"] == obj_name:
                    return const
        return None

    def __parse_epoch(self):
        # Convert TLE epoch string to datetime object
        return datetime.strptime(self.tle["EPOCH"], "%Y-%m-%dT%H:%M:%S.%f")

    def __apply_constellation_specific_settings(self):
        """
        Apply any constellation-specific adjustments based on detected constellation.
        """
        if self.const == 'GPS':
            # GPS satellites typically have circular orbits (low eccentricity)
            self.inclination_limit = 55  # Example, typical inclination range
        elif self.const == 'GAL':
            # Galileo satellites have medium eccentricity orbits
            self.inclination_limit = 56  # Example for Galileo
        elif self.const == 'GLO':
            # GLONASS uses medium earth orbits with slightly different configurations
            self.inclination_limit = 65  # Example for GLONASS
        elif self.const == 'BEI':
            # BeiDou has a mixture of MEO, IGSO, and GEO satellites
            self.inclination_limit = 55  # Example for BeiDou IGSO

    def time_since_epoch(self, time):
        # Calculate time difference in seconds between given time and TLE epoch
        delta = time - self.epoch
        return delta.total_seconds()

    def mean_anomaly(self, delta_t):
        # Calculate mean anomaly at time delta_t (in seconds) since epoch
        n = self.tle["MEAN_MOTION"] * (2 * math.pi) / 86400  # Convert rev/day to rad/s
        M0 = math.radians(self.tle["MEAN_ANOMALY"])  # Initial mean anomaly in radians
        return (M0 + n * delta_t) % (2 * math.pi)  # Mean anomaly at time t

    def solve_kepler(self, M, e, tol=1e-8):
        # Solve Kepler's equation E - e*sin(E) = M for eccentric anomaly E
        E = M  # Initial guess
        while True:
            E_next = E - (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
            if abs(E_next - E) < tol:
                break
            E = E_next
        return E

    def true_anomaly(self, E, e):
        # Compute true anomaly from eccentric anomaly
        return 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))

    def semi_major_axis(self):
        # Compute the semi-major axis using Kepler's 3rd law
        n = self.tle["MEAN_MOTION"] * (2 * math.pi) / 86400  # rad/s
        return (self.mu / n ** 2) ** (1 / 3)  # semi-major axis in km

    def position_orbital_plane(self, r, nu):
        # Compute position in the orbital plane (x', y') given radius r and true anomaly nu
        return r * math.cos(nu), r * math.sin(nu)

    def rotate_to_eci(self, x_orb, y_orb, nu):
        # Apply three rotations: Argument of perigee, Inclination, RAAN
        i = math.radians(self.tle["INCLINATION"])
        Omega = math.radians(self.tle["RA_OF_ASC_NODE"])
        omega = math.radians(self.tle["ARG_OF_PERICENTER"])

        # First rotate by argument of perigee
        x_p = x_orb * math.cos(omega) - y_orb * math.sin(omega)
        y_p = x_orb * math.sin(omega) + y_orb * math.cos(omega)

        # Second rotate by inclination
        y_p_incl = y_p * math.cos(i)
        z_p_incl = y_p * math.sin(i)

        # Third rotate by RAAN
        x_eci = x_p * math.cos(Omega) - y_p_incl * math.sin(Omega)
        y_eci = x_p * math.sin(Omega) + y_p_incl * math.cos(Omega)
        z_eci = z_p_incl

        return x_eci, y_eci, z_eci

    def eci_to_geo(self, eci_coordinates):
        # Convert ECI coordinates to latitude, longitude, and altitude
        x, y, z = eci_coordinates
        r_earth = 6371  # Earth's radius (km)

        # Calculate the distance from the Earth's center
        r = math.sqrt(x**2 + y**2 + z**2)
        
        # Calculate the altitude above the Earth's surface
        altitude = r - r_earth

        # Calculate latitude and longitude
        latitude = math.degrees(math.asin(z / r))  # Latitude in degrees
        longitude = math.degrees(math.atan2(y, x))  # Longitude in degrees

        return (latitude, longitude, altitude)

    def compute_position(self, time):
        # Main method to compute satellite position in ECI coordinates at a given time
        delta_t = self.time_since_epoch(time)  # Time since TLE epoch in seconds
        a = self.semi_major_axis()  # Semi-major axis in km
        e = self.tle["ECCENTRICITY"]  # Orbital eccentricity
        M = self.mean_anomaly(delta_t)  # Mean anomaly at time t
        E = self.solve_kepler(M, e)  # Eccentric anomaly
        nu = self.true_anomaly(E, e)  # True anomaly
        r = a * (1 - e * math.cos(E))  # Orbital radius at true anomaly nu
        x_orb, y_orb = self.position_orbital_plane(r, nu)  # Position in orbital plane
        return self.rotate_to_eci(x_orb, y_orb, nu)  # Position in ECI coordinates









def plot_satellite_position(lat, lon):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())

    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
    ax.scatter(lon, lat, color='blue', s=100, zorder=5)
    plt.title('Current Position of the Satellite')
    plt.show()




tel =  {
"OBJECT_NAME": "GPS BIIF-12 (PRN 32)",
"OBJECT_ID": "2016-007A",
"EPOCH": "2024-10-23T01:53:25.046304",
"MEAN_MOTION": 2.005556,
"ECCENTRICITY": 0.008137,
"INCLINATION": 55.1816,
"RA_OF_ASC_NODE": 113.3676,
"ARG_OF_PERICENTER": 238.4766,
"MEAN_ANOMALY": 120.7561,
"EPHEMERIS_TYPE": 0,
"CLASSIFICATION_TYPE": "U",
"NORAD_CAT_ID": 41328,
"ELEMENT_SET_NO": 999,
"REV_AT_EPOCH": 6376,
"BSTAR": 0,
"MEAN_MOTION_DOT": 3.3e-7,
"MEAN_MOTION_DDOT": 0
}
satellite = Satellite(tel)
print(satellite.const)
print(satellite.epoch)
in_time = datetime(2024, 10, 24, 18, 20, 0)  # Example time
# in_time = datetime.datetime.now(datetime.timezone.utc)
print(in_time)
print(satellite.compute_position(in_time))

eci_position = satellite.compute_position(in_time)
geo_coordinates = satellite.eci_to_geo(eci_position)
print(geo_coordinates)


# Current time
# in_time = datetime.now(datetime.timezone.utc)  # Use current UTC time
print(f"Current Time: {in_time}")

# Compute satellite's position
eci_position = satellite.compute_position(in_time)
geo_coordinates = satellite.eci_to_geo(eci_position)
print(f"Geodetic Coordinates: Latitude = {geo_coordinates[0]:.4f}, Longitude = {geo_coordinates[1]:.4f}, Altitude = {geo_coordinates[2]:.4f} km")

# # Plotting
# plt.figure(figsize=(10, 6))
# m = Basemap(projection='merc', llcrnrlat=-60, urcrnrlat=85,
#             llcrnrlon=-180, urcrnrlon=180, resolution='i')

# # Draw coastlines and countries
# m.drawcoastlines()
# m.drawcountries()

# Plot the satellite position
lat = geo_coordinates[0]
lon = geo_coordinates[1]
print(lat, lon)



# Use the function to plot
plot_satellite_position(geo_coordinates[0], geo_coordinates[1])

# m.plot(lon, lat, 'bo', markersize=10)  # Plot satellite position
# plt.text(lon, lat, 'Satellite', fontsize=12, ha='right', color='blue')

# plt.title('Current Position of the Satellite')
# plt.show()
