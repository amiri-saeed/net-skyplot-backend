import requests
from pprint import pprint as pp

BASE_URL = "http://127.0.0.1:5000"


# /dems
dems_payload = {
    "receivers": [
        {"coordinates": {"latitude": 37.7749, "longitude": -122.4194}},
        {"coordinates": {"latitude": 37.7859, "longitude": -122.4364}},
        # {"coordinates": {"latitude": 45.072841, "longitude": 7.682311}}
        ]
}

resp = requests.post(BASE_URL+'/dems', json=dems_payload)
pp(resp.json())



# /paln
plan_payload = {
  "start_datetime": "2025-03-02T12:00:00Z",
  "duration_hours": 4,
  "dem": {"type": "SRTMGL3", "source": "ot"},
  "constellations": [
    "GPS",
    "GALILEO",
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
          "height": 12
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

resp = requests.post(BASE_URL+'/plan', json=plan_payload)
pp(resp.json())