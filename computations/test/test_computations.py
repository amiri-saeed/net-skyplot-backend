import requests
from pprint import pprint as pp
import json

BASE_URL = "http://127.0.0.1:5002"


# # /metrics
# metrics_payload  = {
#   "start_datetime": "2025-02-24T10:00:00Z",
#   "duration_hours": 2,
#   "dem": {"type": "SRTMGL3", "source": "ot"},
#   "constellations": ["GPS", "GLONASS"],
#   "receivers": [
#     {
#       "id": "receiver_1",
#       "role": "base",
#       "coordinates": {
#         "latitude" :  40.712776,
#         "longitude": -74.005974,
#         "height"   :  15.0
#       },
#       "obstacles": [
#         {
#           "vertices": [
#             {"latitude": 40.713000, "longitude": -74.005500},
#             {"latitude": 40.713200, "longitude": -74.005700},
#             {"latitude": 40.713100, "longitude": -74.005900}
#           ],
#           "height": 12
#         }
#       ]
#     },
#     {
#       "id": "receiver_2",
#       "role": "rover",
#       "coordinates": {
#         "latitude" :  42.712776,
#         "longitude": -71.005974,
#         "height"   :  12.0
#       },
#       "obstacles": [
#         # {
#         #   "vertices": [
#         #     {"latitude": 40.713000, "longitude": -74.005500},
#         #     {"latitude": 40.713200, "longitude": -74.005700},
#         #     {"latitude": 40.713100, "longitude": -74.005900}
#         #   ],
#         #   "height": 15
#         # }
#       ]
#     }
#   ]
# }

metrics_payload = {
  "start_datetime": "2025-03-02T12:00:00Z",
  "duration_hours": 24,
  "dem": {"type": "SRTMGL1", "source": "ot"},
  "constellations": [
    "GPS",
    # "GALILEO",
    # "GLONASS",
    # "BEIDOU"
    ],
  "receivers": [
    {
      "id": "receiver_1",
      "role": "base",
      "coordinates": {
        "latitude": 45.0703,
        "longitude": 7.6869,
        "height": 2.0
      },
      "obstacles": [
        {
          "vertices": [
            {"latitude": 45.07106612022467, "longitude": 7.686492204666139},
            {"latitude": 45.070535726562476, "longitude": 7.686148881912232},
            {"latitude": 45.07080092400884, "longitude": 7.685312032699586},
            {"latitude": 45.071338892191505, "longitude": 7.685687541961671},
            {"latitude": 45.07106612022467, "longitude": 7.686492204666139}
          ],
          "height": 100
        }
      ]
    },
    {
      "id": "receiver_2",
      "role": "rover",
      "coordinates": {
        "latitude": 45.070793761326115,
        "longitude": 7.683906555175781,
        "height": 1.0
      },
      "obstacles": []
    }
  ],
  "application": "differential_gnss"
}
resp = requests.post(BASE_URL+'/metrics', json=metrics_payload)
pp(resp.json())

# with open('response.json', 'w') as f:
#     json.dump(resp.json(), f, indent=4)







# from sgp4.api import Satrec, WGS84


# # # ISS TLE for testing
# # tle = [
# #     "1 25544U 98067A   23298.52534722  .00016717  00000-0  10270-3 0  9999",
# #     "2 25544  51.6416 208.4053 0007003  92.3350 267.7844 15.49970385414828"
# # ]

# # tle = [
# # "1 24876U 97035A   25053.56631748 -.00000002  00000+0  00000+0 0  9997",
# # "2 24876  55.7583 117.3378 0086980  54.1368 306.7307  2.00562570202333"
# # ]

# # BEIDOU-3 M28 (C50)
# tle = [
# "1 58654U 23207A   25052.72360418 -.00000001  00000+0  00000+0 0  9997",
# "2 58654  55.3223  78.5668 0004711 287.1849  72.8245  1.86232083  7897"
# ]

# # Initialize satellite
# sat = Satrec.twoline2rv(tle[0], tle[1], WGS84)

# # Propagate to a specific time
# jd, fr = 2460258.5, 0.0  # Example time
# # jd, fr = 27449.5, 0.3333333333333333
# error_code, r, v = sat.sgp4(jd, fr)

# if error_code != 0:
#     print(f"Error: {sat.error_message}")
# else:
#     print(f"Position: {r} km")









# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# # Function to convert ECEF to Lat/Lon
# def ecef_to_latlon(x, y, z):
#     a = 6378.137  # Earth's semi-major axis in km
#     e2 = 0.00669437999014  # Earth's eccentricity squared

#     lon = np.degrees(np.arctan2(y, x))
#     p = np.sqrt(x**2 + y**2)
#     lat = np.degrees(np.arctan2(z, p * (1 - e2)))  # Approximate formula

#     return lat, lon

# # ECEF coordinates (km)
# ecef_positions = [
#     (-4044.5634300087745, -15190.929936523791, 21970.20080848693),
#     (2715.993567006888, -16108.89455960682, 21422.73181308945),
#     (9297.418716695958, -15968.665445717841, 19467.92347607363),
#     (15260.451858395087, -14767.91860286088, 16220.198690084231),
#     (20195.502677576198, -12573.496235518624, 11881.175307429678),
#     (23750.575252441064, -9520.833737544854, 6731.397429103842),
#     (25658.702025905637, -5808.749795304137, 1115.8189627740558),
#     (25762.361579349363, -1688.839774773586, -4577.32880063514)
# ]

# # Convert ECEF to Lat/Lon
# latitudes, longitudes = zip(*[ecef_to_latlon(x, y, z) for x, y, z in ecef_positions])

# # Plot on a world map
# fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
# ax.set_global()
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.BORDERS, linestyle=':')

# # Plot satellite path
# ax.scatter(longitudes, latitudes, color='red', marker='o', label="Satellite Positions")
# ax.plot(longitudes, latitudes, color='blue', linestyle='--', label="Satellite Path")

# plt.legend()
# plt.title("Satellite Path Projection on 2D Map")
# plt.show()





# import json 
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D



# source = "ot"

# dem_payload = {"coordinates": [{"lat": 45.066301, "lon": 7.668238}],
#                 "selected_source": f"{source}",
#                 "dem_type": "SRTMGL1"
# }

# resp = requests.post("http://localhost:5001/dem", json=dem_payload)

# with open(f'{source}.json', 'w') as f:
#     json.dump(resp.json(), f)


# # Load JSON data
# with open(f"{source}.json", "r") as file:  # Replace with your filename
#     data = json.load(file)

# # Extract elevation data (assumed to be a 2D list)
# elevation = np.array(data["elevation"])
# print(elevation.shape)

# # Generate X and Y coordinate grids
# x = np.arange(elevation.shape[1])
# y = np.arange(elevation.shape[0])
# X, Y = np.meshgrid(x, y)

# # Create 3D surface plot
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(X, Y, elevation, cmap="terrain", edgecolor="k", linewidth=0.1)

# # Labels and title
# ax.set_xlabel("Longitude Index")
# ax.set_ylabel("Latitude Index")
# ax.set_zlabel("Elevation (m)")
# ax.set_title("3D Surface Plot of Elevation Data")

# # Show plot
# # plt.show()
# plt.savefig(f"{source}.png", dpi=300)







# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.collections import LineCollection
# import matplotlib.dates as mdates
# from datetime import datetime

# def plot_skyplot(skyplot_data):
#     # Create polar plot
#     fig = plt.figure(figsize=(12, 12))
#     ax = fig.add_subplot(111, projection='polar')
    
#     # Configure plot settings
#     ax.set_theta_zero_location('N')
#     ax.set_theta_direction(-1)
#     ax.set_ylim(0, 90)
#     ax.set_yticks(range(0, 90+1, 15))
#     ax.set_yticklabels([str(90 - y) for y in range(0, 90+1, 15)])
#     ax.grid(True, alpha=0.5)
    
#     # Constellation styling
#     constellation_colors = {
#         'GPS': 'blue',
#         'GALILEO': 'red',
#         'GLONASS': 'green',
#         'BEIDOU': 'orange'
#     }
    
#     # Collect all timestamps and initialize colormap
#     all_times = []
#     for sat in skyplot_data['satellites']:
#         all_times.extend([datetime.fromisoformat(p['time']) for p in sat['trajectory'] if p['visible']])
    
#     min_time = min(all_times)
#     max_time = max(all_times)
#     norm = mdates.date2num(all_times)
#     time_norm = plt.Normalize(norm.min(), norm.max())
#     cmap = plt.cm.viridis
    
#     # Plot each satellite's trajectory
#     for satellite in skyplot_data['satellites']:
#         # Filter and sort visible points
#         visible_points = [p for p in satellite['trajectory'] if p['visible']]
#         visible_points.sort(key=lambda x: datetime.fromisoformat(x['time']))
        
#         if len(visible_points) < 2:
#             continue  # Skip satellites with insufficient data
            
#         # Convert coordinates
#         azimuths = np.deg2rad([p['azimuth'] for p in visible_points])
#         elevations = [90 - p['elevation'] for p in visible_points]
#         times = [datetime.fromisoformat(p['time']) for p in visible_points]
        
#         # Create line segments with time-based coloring
#         points = np.array([azimuths, elevations]).T.reshape(-1, 1, 2)
#         segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
#         # Convert times to numerical values
#         time_values = mdates.date2num(times)
#         segment_colors = cmap(time_norm(time_values[:-1]))
        
#         # Create line collection
#         lc = LineCollection(
#             segments,
#             colors=segment_colors,
#             linewidth=1.5,
#             alpha=0.7,
#             label=satellite['constellation']
#         )
#         ax.add_collection(lc)
        
#         # Add start/end markers
#         start_color = cmap(time_norm(time_values[0]))
#         end_color = cmap(time_norm(time_values[-1]))
#         ax.scatter(azimuths[0], elevations[0], color=start_color, s=50, edgecolor='w', zorder=3)
#         ax.scatter(azimuths[-1], elevations[-1], color=end_color, s=50, marker='s', edgecolor='w', zorder=3)
    
#     # Add colorbar
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=time_norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax, pad=0.1)
#     cbar.set_label('Time', rotation=270, labelpad=20)
#     cbar.ax.yaxis_date()
#     cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
#     # Add legend
#     handles, labels = ax.get_legend_handles_labels()
#     unique_labels = []
#     unique_handles = []
#     for h, l in zip(handles, labels):
#         if l not in unique_labels:
#             unique_labels.append(l)
#             unique_handles.append(h)
#     ax.legend(
#         unique_handles,
#         unique_labels,
#         title='Constellations',
#         loc='upper right',
#         bbox_to_anchor=(1.25, 1))
    
#     plt.title("Satellite Skyplot with Time Evolution\n", pad=20, fontsize=14)
#     plt.tight_layout()
#     plt.show()

# # Example usage with your data
# if __name__ == "__main__":
#     # Load your skyplot data here
#     with open("response.json", "r") as file:  # Replace with your filename
#         data = json.load(file)
#     skyplot_data = data["receivers"][0]["skyplot_data"]
    
#     plot_skyplot(skyplot_data)