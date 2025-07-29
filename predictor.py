
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# --- 1. Load All Models and Data ---
print("Loading all prediction models and historical data...")
try:
    regr_model = joblib.load('corner_regression_model.joblib')
    model_features = joblib.load('model_features.joblib')
    historical_data = pd.read_csv('top_5_leagues_cleaned_data.csv', parse_dates=['MatchDate'])
    
    # Load all classification models
    corner_lines = [8.5, 9.5, 10.5, 11.5, 12.5]
    class_models = {}
    for line in corner_lines:
        model_filename = f'corner_classifier_o{str(line).replace(".", "")}.joblib'
        class_models[line] = joblib.load(model_filename)
        
    print("✅ All models and data loaded successfully.")

except FileNotFoundError as e:
    print(f"❌ Error loading a model or data file: {e}")
    regr_model = class_models = model_features = historical_data = None

# --- 2. Self-Contained Feature Calculation Functions ---

def get_team_form(team_name, match_date, historical_data):
    """
    Calculates the rolling 5-game average stats for a single team before a given match date.
    """
    team_matches = historical_data[
        ((historical_data['HomeTeam'] == team_name) | (historical_data['AwayTeam'] == team_name)) &
        (historical_data['MatchDate'] < match_date)
    ].tail(5)

    if len(team_matches) < 3: # Require at least 3 historical matches
        return None

    stats = []
    for index, row in team_matches.iterrows():
        is_home = row['HomeTeam'] == team_name
        stats.append({
            'corners_for': row['HomeCorners'] if is_home else row['AwayCorners'],
            'corners_against': row['AwayCorners'] if is_home else row['HomeCorners'],
            'shots_for': row['HomeShots'] if is_home else row['AwayShots'],
            'shots_against': row['AwayShots'] if is_home else row['HomeShots'],
            'shots_target_for': row['HomeTarget'] if is_home else row['AwayTarget'],
            'goals_for': row['FTHome'] if is_home else row['FTAway'],
            'goals_against': row['FTAway'] if is_home else row['FTHome'],
        })
    
    form_df = pd.DataFrame(stats)
    return form_df.mean().to_dict()

def get_h2h_history(home_team, away_team, match_date, historical_data):
    """
    Calculates the average total corners from the last 3 head-to-head matches.
    """
    h2h_matches = historical_data[
        (historical_data['MatchDate'] < match_date) &
        (
            ((historical_data['HomeTeam'] == home_team) & (historical_data['AwayTeam'] == away_team)) |
            ((historical_data['HomeTeam'] == away_team) & (historical_data['AwayTeam'] == home_team))
        )
    ].tail(3)

    if h2h_matches.empty:
        return None
    
    return (h2h_matches['HomeCorners'] + h2h_matches['AwayCorners']).mean()

# --- 3. Main Prediction Function ---
def make_multi_line_prediction(match_data):
    if not all([regr_model, class_models, model_features, isinstance(historical_data, pd.DataFrame)]):
        return {"error": "Models or historical data not loaded. Cannot make prediction."}

    home_team = match_data['home_team']
    away_team = match_data['away_team']
    league_name = match_data.get('league_name', 'Premier League')
    match_date = datetime.now()

    home_form = get_team_form(home_team, match_date, historical_data)
    away_form = get_team_form(away_team, match_date, historical_data)

    if not home_form or not away_form:
        return {"error": f"Not enough recent games for {home_team} or {away_team} to make a prediction."}

    h2h_avg_corners = get_h2h_history(home_team, away_team, match_date, historical_data)
    
    league_data = historical_data[historical_data['LeagueName'] == league_name]
    if league_data.empty:
        return {"error": f"No historical data for league: {league_name}"}
    
    league_avg_corners = (league_data['HomeCorners'].mean() + league_data['AwayCorners'].mean()) / 2

    if pd.isna(h2h_avg_corners):
        h2h_avg_corners = league_avg_corners

    # Build the feature vector for the model
    feature_dict = {
        'home_corners_for_L5': home_form['corners_for'],
        'home_corners_against_L5': home_form['corners_against'],
        'away_corners_for_L5': away_form['corners_for'],
        'away_corners_against_L5': away_form['corners_against'],
        'home_shots_for_L5': home_form['shots_for'],
        'away_shots_for_L5': away_form['shots_for'],
        'home_attack_strength': home_form['corners_for'] / league_avg_corners,
        'home_defense_strength': home_form['corners_against'] / league_avg_corners,
        'away_attack_strength': away_form['corners_for'] / league_avg_corners,
        'away_defense_strength': away_form['corners_against'] / league_avg_corners,
        'h2h_avg_corners_L3': h2h_avg_corners,
        'OddHome': match_data['odd_home'],
        'OddDraw': match_data['odd_draw'],
        'OddAway': match_data['odd_away']
    }

    prediction_df = pd.DataFrame([feature_dict], columns=model_features)

    # Make Predictions
    predicted_corners = regr_model.predict(prediction_df)[0]
    
    probabilities = {}
    for line, model in class_models.items():
        prob_array = model.predict_proba(prediction_df)[0]
        probabilities[f'over_{str(line).replace(".", "_")}'] = round(float(prob_array[1]), 4)

    return {
        "predicted_total_corners": round(float(predicted_corners), 2),
        "probabilities": probabilities
    }