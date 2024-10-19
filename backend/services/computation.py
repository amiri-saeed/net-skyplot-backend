# services/core_computation.py
from services.data_integration import fetch_data_from_source
import json
import os

def perform_computation(lon, lat, start, end, polygon, height):
    """
    Perform the core computation needed for the plan endpoint.
    This may involve fetching additional data, processing input parameters, etc.
    """
    # Example of fetching data using the data_integration service
    # external_data = fetch_data_from_source("almanac")

    external_data = json.load(open("services/data_gps.json"))["DATA"]

    if not external_data:
        return {"error": "Failed to fetch necessary data for computation"}

    # Placeholder for actual computation logic
    # This example simply returns a mock result combining the inputs and fetched data
    result = {
        "location": {"lon": lon, "lat": lat},
        "time_window": {"start": start, "end": end},
        "obstacle": {"polygon": polygon, "height": height},
        "external_data": external_data
    }

    return result
