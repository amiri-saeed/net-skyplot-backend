# services/data_integration.py
import requests
import json
import os


def fetch_data_from_source(_type="almanac"):

    
    if _type == "almanac":
        with open("services/config.json", "r") as f:
            conf = json.load(f)[_type]
            for alm_data in conf:
                url = alm_data["url"]
                response = fetch_almanac(url)
                if response:
                    data = response
                else:
                    return None

    if _type == "dem":
        pass

    return data



def fetch_almanac(url):
    response = requests.get(url)
    if response.status_code != 200:
        response = None
    return response.json()