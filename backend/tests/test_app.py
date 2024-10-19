import unittest
import requests

class TestFlaskApp(unittest.TestCase):

    BASE_URL = "http://127.0.0.1:5000"

    def test_plan_route(self):
        # Define the test payload
        payload = {
            "lon": 12.34,
            "lat": 56.78,
            "start": "2024-10-18T08:00:00",
            "end": "2024-10-18T09:00:00",
            "polygon": [
                [12.34, 56.78],
                [12.35, 56.79],
                [12.36, 56.77]
            ],
            "height": 10.5
        }

        # Send the POST request
        response = requests.post(f"{self.BASE_URL}/plan", json=payload)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)

        # Check the response data
        response_data = response.json()
        self.assertIn("message", response_data)
        self.assertEqual(response_data["message"], "Parameters received successfully")
        self.assertEqual(response_data["lon"], payload["lon"])
        self.assertEqual(response_data["lat"], payload["lat"])
        self.assertEqual(response_data["start"], payload["start"])
        self.assertEqual(response_data["end"], payload["end"])
        self.assertEqual(response_data["polygon"], payload["polygon"])
        self.assertEqual(response_data["height"], payload["height"])

    # Additional tests can be added here

if __name__ == '__main__':
    unittest.main()
