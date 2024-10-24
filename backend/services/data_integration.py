# services/data_integration.py
import requests
import json
import os
import logging


class Almanac:

    def __init__(self):
        pass


    def fetch_almanac(self, url):
        response = requests.get(url)
        if response.status_code != 200:
            response = None
        return response.json()


    def fetch():        
        with open("services/config.json", "r") as f:
            conf = json.load(f)["almanac"]
            for alm_data in conf:
                url = alm_data["url"]
                response = self.fetch_almanac(url)
                if response:
                    data = response
                else:
                    return None


        return data



class Constellations:

    def __init__(self):
        self.conf_path = "services/config.json"
        self.conf = json.load(open(self.conf_path))
        self.sat_msk = {
            "OBJECT_NAME": "",
            "OBJECT_ID": "",
            "NORAD_CAT_ID": "",
        }

    def _prune_dict(self, dct):
        result = {}
        for k, v in self.sat_msk.items():
            result[k] = dct[k]
        return result

    def write_sats(self, const, sats):
        self.conf["constellations"][const]["sats"] = sats
        return json.dump(self.conf, open(self.conf_path, 'w'))

    def update_sats(self):
        for const in list(self.conf["constellations"]):
            sats = list()
            url = self.conf["constellations"][const]["url"]
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    for resp in response.json():
                        sat = self._prune_dict(resp)
                        sats.append(sat)
            except Exception as e:
                logging.info(e)

            logging.info(const, sats)
            self.write_sats(const, sats)