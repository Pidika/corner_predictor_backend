#last working version 
# import os
# from flask import Flask, send_file, jsonify, request
# from flask_cors import CORS
# from dotenv import load_dotenv
# import pandas as pd

# # UPDATED: Import the new multi-line prediction function
# from predictor import make_multi_line_prediction
# # from odds_fetcher import fetch_upcoming_matches 

# load_dotenv()

# app = Flask(__name__)
# CORS(app)

# # --- API Endpoints ---

# @app.route('/')
# def index():
#     return "Welcome to the Corner Kick Predictor API!"


# @app.route('/api/upcoming-matches', methods=['GET'])
# def get_upcoming_matches():
#     try:
#         file_path = os.path.join('public', 'upcoming_features.json')
#         return send_file(file_path, mimetype='application/json', as_attachment=True)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# @app.route('/api/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     required_fields = ['home_team', 'away_team', 'odd_home', 'odd_draw', 'odd_away']
#     if not all(field in data for field in required_fields):
#         return jsonify({"error": "Missing required fields in request body"}), 400

#     # UPDATED: Call the new multi-line prediction function
#     prediction = make_multi_line_prediction(data)

#     if "error" in prediction:
#         return jsonify(prediction), 404

#     return jsonify(prediction)

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)


import os
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

from predictor import make_multi_line_prediction

load_dotenv()

# Serve React App from a 'build' folder in production
app = Flask(__name__, static_folder='build', static_url_path='/')
CORS(app)

@app.route('/api/upcoming-matches', methods=['GET'])
def get_upcoming_matches():
   
    try:
       
        return send_from_directory('public', 'upcoming_matches.json')
    except FileNotFoundError:
        return jsonify({"error": "Upcoming matches file not found. Please run the odds_fetcher.py script."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    required_fields = ['home_team', 'away_team', 'odd_home', 'odd_draw', 'odd_away']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields in request body"}), 400

    prediction = make_multi_line_prediction(data)

    if "error" in prediction:
        return jsonify(prediction), 404

    return jsonify(prediction)

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)