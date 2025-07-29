import os
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

from predictor import make_multi_line_prediction
from odd_fetcher import fetch_and_save_upcoming_matches

load_dotenv()

app = Flask(__name__, static_folder='build', static_url_path='/')
CORS(app)

# Secret key for triggering the cron job
CRON_SECRET = os.getenv('CRON_SECRET')

@app.route('/api/upcoming-matches', methods=['GET'])
def get_upcoming_matches():
    """
    Serves the pre-fetched upcoming matches from a static JSON file.
    """
    try:
        # The JSON file should be in a 'public' directory within your backend folder
        return send_from_directory('public', 'upcoming_matches.json')
    except FileNotFoundError:
        return jsonify({"error": "Upcoming matches file not found. Please run the odds_fetcher.py script."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Receives match data and returns a prediction from our trained models.
    """
    data = request.get_json()
    required_fields = ['home_team', 'away_team', 'odd_home', 'odd_draw', 'odd_away']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields in request body"}), 400

    prediction = make_multi_line_prediction(data)

    if "error" in prediction:
        return jsonify(prediction), 404

    return jsonify(prediction)

@app.route('/api/trigger-odds-fetch', methods=['POST'])
def trigger_odds_fetch():
    """
    A secure endpoint to trigger the daily odds fetching script,
    callable by an external cron job service.
    """
    auth_header = request.headers.get('Authorization')
    if not CRON_SECRET or auth_header != f'Bearer {CRON_SECRET}':
        return jsonify({"error": "Unauthorized"}), 401
    
    print("Cron job triggered. Fetching upcoming matches...")
    result = fetch_and_save_upcoming_matches()
    return jsonify(result)

# Serve React App (for production)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """
    Serves the React application's static files.
    This handles all frontend routing.
    """
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
