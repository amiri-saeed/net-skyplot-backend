

'''
End Points
    /plan
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

'''


# app/routes.py
from flask import Blueprint, request, jsonify
from services.computation import perform_computation

# Create a blueprint for the app
main = Blueprint('main', __name__)

@main.route('/plan', methods=['POST'])
def plan_route():
    try:
        # Parse the JSON payload
        data = request.get_json()
        print(data)

        # Extract required parameters from the payload
        lon = data.get('lon')
        lat = data.get('lat')
        start = data.get('start')
        end = data.get('end')
        polygon = data.get('polygon')  # Expecting a list of corner points (lat, lon)
        height = data.get('height')

        # Validate the received data
        if lon is None or lat is None or start is None or end is None:
            return jsonify({"error": "Missing required parameters"}), 400

        if not polygon or not isinstance(polygon, list) or height is None:
            return jsonify({"error": "Missing or invalid obstacle data"}), 400

        # Perform the core computation using the provided data
        result = perform_computation(lon, lat, start, end, polygon, height)

        # Check if there was an error in the computation
        if "error" in result:
            return jsonify(result), 500

        return jsonify(result), 200

    except Exception as e:
        # Handle any unexpected errors
        return jsonify({"error": str(e)}), 500
