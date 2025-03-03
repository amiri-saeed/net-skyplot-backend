import requests
from pprint import pprint as pp
import time


# # /dem_availability
# payload = {
#     "receivers":[
#         {"coordinates": { "latitude": 59.065219, "longitude": 10.676745}},
#         {"coordinates": { "latitude": 45.072841, "longitude": 7.682311}},
#     ]
# }
# response = requests.post("http://localhost:5001/dem_availability", json=payload)
# pp(response.json())


# now = time.time()
# # /dem
# payload = {
#     "coordinates": [
#         # {"lat": 45.065219, "lon": 7.676745},
#         # {"lat": 45.072841, "lon": 7.682311}
#         # {"lat": 46.06382, "lon": 8.30178}
#         {"lat": 44.89661, "lon": 7.86572}
#     ],
#     "selected_source": "ot",
#     "dem_type": "SRTMGL3"
# }
# response = requests.post("http://localhost:5001/dem", json=payload)
# print(response.json())
# print(time.time() - now)


# /alm
response = requests.get("http://localhost:5001/alm")
pp(response.json())


# #  constellations
# response = requests.get("http://localhost:5001/constellations")
# pp(response.json())


# # refresh constellations
# response = requests.post("http://localhost:5001/constellations/update")
# print(response.json())
