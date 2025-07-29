# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, accuracy_score
# import joblib
# import warnings

# # Suppress a common future warning from pandas/numpy to keep the output clean
# warnings.filterwarnings('ignore', category=FutureWarning)

# def feature_engineering(df):
#     """
#     Engineers features for the model based on historical data.
#     This function now calculates rolling averages, strength ratings, and H2H stats.
#     """
#     print("Starting feature engineering...")
    
#     # Ensure data is sorted by date for accurate rolling calculations
#     df = df.sort_values('MatchDate').copy()
    
#     # --- Part 1: Rolling Averages (Team Form) ---
#     # Create a DataFrame from each team's perspective for easier calculation
#     match_list = []
#     for index, row in df.iterrows():
#         # Add home team perspective
#         match_list.append({
#             'LeagueName': row['LeagueName'], 'team': row['HomeTeam'], 'opponent': row['AwayTeam'], 'date': row['MatchDate'],
#             'goals_for': row['FTHome'], 'goals_against': row['FTAway'], 'shots_for': row['HomeShots'],
#             'shots_against': row['AwayShots'], 'shots_target_for': row['HomeTarget'], 'shots_target_against': row['AwayTarget'],
#             'corners_for': row['HomeCorners'], 'corners_against': row['AwayCorners'],
#         })
#         # Add away team perspective
#         match_list.append({
#             'LeagueName': row['LeagueName'], 'team': row['AwayTeam'], 'opponent': row['HomeTeam'], 'date': row['MatchDate'],
#             'goals_for': row['FTAway'], 'goals_against': row['FTHome'], 'shots_for': row['AwayShots'],
#             'shots_against': row['HomeShots'], 'shots_target_for': row['AwayTarget'], 'shots_target_against': row['HomeTarget'],
#             'corners_for': row['AwayCorners'], 'corners_against': row['HomeCorners'],
#         })
        
#     team_stats_df = pd.DataFrame(match_list).sort_values(by=['team', 'date'])
#     team_stats_df = team_stats_df.reset_index(drop=True)

#     # --- Use .transform() for a more robust rolling average calculation ---
#     # This method is safer and avoids potential index alignment issues.
#     stats_to_average = [
#         'goals_for', 'goals_against', 'shots_for', 'shots_against',
#         'shots_target_for', 'shots_target_against', 'corners_for', 'corners_against'
#     ]
    
#     for stat in stats_to_average:
#         # Group by team, then apply a lambda function that calculates the rolling mean and shifts it
#         rolling_avg = team_stats_df.groupby('team')[stat].transform(
#             lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
#         )
#         team_stats_df[f'avg_{stat}_L5'] = rolling_avg

#     # Merge rolling averages back into the main dataframe
#     df = df.merge(
#         team_stats_df[['team', 'date'] + [f'avg_{s}_L5' for s in stats_to_average]],
#         left_on=['HomeTeam', 'MatchDate'], right_on=['team', 'date'], how='left'
#     ).rename(columns={f'avg_{s}_L5': f'home_{s}_L5' for s in stats_to_average})

#     df = df.merge(
#         team_stats_df[['team', 'date'] + [f'avg_{s}_L5' for s in stats_to_average]],
#         left_on=['AwayTeam', 'MatchDate'], right_on=['team', 'date'], how='left', suffixes=('', '_away')
#     ).rename(columns={f'avg_{s}_L5': f'away_{s}_L5' for s in stats_to_average})

#     # Correctly drop the columns created by the merge operations
#     df = df.drop(columns=['team', 'date', 'team_away', 'date_away'])


#     # --- Part 2: Attacking & Defensive Strength Ratings ---
#     # Calculate league averages for corners
#     league_avg_stats = df.groupby('LeagueName').agg(
#         league_avg_corners=('HomeCorners', lambda x: (x.sum() + df.loc[x.index, 'AwayCorners'].sum()) / (2 * len(x)))
#     ).reset_index()
    
#     df = df.merge(league_avg_stats, on='LeagueName', how='left')

#     # Calculate strength ratings by comparing team's form to league average
#     df['home_attack_strength'] = df['home_corners_for_L5'] / df['league_avg_corners']
#     df['home_defense_strength'] = df['home_corners_against_L5'] / df['league_avg_corners']
#     df['away_attack_strength'] = df['away_corners_for_L5'] / df['league_avg_corners']
#     df['away_defense_strength'] = df['away_corners_against_L5'] / df['league_avg_corners']

#     # --- Part 3: Head-to-Head (H2H) Features ---
#     h2h_corners = []
#     df['total_corners_temp'] = df['HomeCorners'] + df['AwayCorners'] # Temp column for H2H calc
    
#     for index, row in df.iterrows():
#         # Find past matches between the two teams
#         past_matches = df[
#             (df['MatchDate'] < row['MatchDate']) &
#             (
#                 ((df['HomeTeam'] == row['HomeTeam']) & (df['AwayTeam'] == row['AwayTeam'])) |
#                 ((df['HomeTeam'] == row['AwayTeam']) & (df['AwayTeam'] == row['HomeTeam']))
#             )
#         ].tail(3) # Get the last 3 H2H games
        
#         if not past_matches.empty:
#             h2h_corners.append(past_matches['total_corners_temp'].mean())
#         else:
#             h2h_corners.append(np.nan) # Use NaN if no H2H history exists
    
#     df['h2h_avg_corners_L3'] = h2h_corners
#     df.drop(columns=['total_corners_temp'], inplace=True)
    
#     # Fill NaN H2H values with the overall league average as a reasonable fallback
#     df['h2h_avg_corners_L3'].fillna(df['league_avg_corners'], inplace=True)

#     # --- Final Cleanup ---
#     # Drop rows where rolling averages couldn't be calculated (first few games of each season)
#     df.dropna(inplace=True)
#     print(f"✅ Feature engineering complete. {len(df)} matches available for training.")
#     return df

# def train_model(input_filepath):
#     """
#     Loads data, engineers features, trains regression and classification models,
#     and saves them to disk.
#     """
#     print(f"Loading data from {input_filepath}...")
#     try:
#         df = pd.read_csv(input_filepath, parse_dates=['MatchDate'])
#     except FileNotFoundError:
#         print(f"❌ Error: The file {input_filepath} was not found.")
#         return
        
#     # Check for 'LeagueName' and create it from 'Division' if it's missing
#     if 'LeagueName' not in df.columns:
#         print("Warning: 'LeagueName' column not found. Creating it from 'Division' column.")
#         if 'Division' in df.columns:
#             leagues_map = {
#                 'E0': 'Premier League', 'SP1': 'La Liga', 'I1': 'Serie A',
#                 'D1': 'Bundesliga', 'F1': 'Ligue 1'
#             }
#             df['LeagueName'] = df['Division'].map(leagues_map)
#             # Drop rows from leagues we are not processing
#             df.dropna(subset=['LeagueName'], inplace=True)
#         else:
#             print("❌ Error: Cannot create 'LeagueName' because 'Division' column is also missing.")
#             return

#     # --- Feature Engineering ---
#     df_featured = feature_engineering(df)

#     # --- Define Targets and Features ---
#     df_featured['total_corners'] = df_featured['HomeCorners'] + df_featured['AwayCorners']
#     CORNER_LINE = 10.5
#     df_featured['over_line'] = (df_featured['total_corners'] > CORNER_LINE).astype(int)

#     # Define the NEW, more powerful feature set based on your plan
#     features = [
#         # Team Form (Rolling Averages)
#         'home_corners_for_L5', 'home_corners_against_L5',
#         'away_corners_for_L5', 'away_corners_against_L5',
#         'home_shots_for_L5', 'away_shots_for_L5',
#         # Strength Ratings
#         'home_attack_strength', 'home_defense_strength',
#         'away_attack_strength', 'away_defense_strength',
#         # H2H
#         'h2h_avg_corners_L3',
#         # Pre-match Betting Odds
#         'OddHome', 'OddDraw', 'OddAway'
#     ]
    
#     X = df_featured[features]
#     y_regr = df_featured['total_corners']
#     y_class = df_featured['over_line']

#     # --- Split Data ---
#     X_train, X_test, y_regr_train, y_regr_test, y_class_train, y_class_test = train_test_split(
#         X, y_regr, y_class, test_size=0.2, random_state=42
#     )
#     print(f"\nData split: {len(X_train)} training samples, {len(X_test)} testing samples.")

#     # --- 1. Train Regression Model (Predicts Total Corners) ---
#     print("\n--- Training Regression Model (XGBoost) ---")
#     regr_model = xgb.XGBRegressor(
#         objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
#         max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
#     )
#     # --- FIX: Removed all early stopping arguments to ensure compatibility ---
#     regr_model.fit(X_train, y_regr_train, verbose=False)
    
#     regr_preds = regr_model.predict(X_test)
#     mae = mean_absolute_error(y_regr_test, regr_preds)
#     print(f"📈 Regression MAE: {mae:.4f} (Model's corner prediction is off by ~{mae:.2f} corners on average)")

#     # --- 2. Train Classification Model (Predicts Over/Under Line) ---
#     print("\n--- Training Classification Model (XGBoost) ---")
#     class_model = xgb.XGBClassifier(
#         objective='binary:logistic', n_estimators=1000, learning_rate=0.05,
#         max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
#         use_label_encoder=False
#     )
#     # --- FIX: Removed all early stopping arguments to ensure compatibility ---
#     class_model.fit(X_train, y_class_train, verbose=False)

#     class_preds = class_model.predict(X_test)
#     accuracy = accuracy_score(y_class_test, class_preds)
#     print(f"🎯 Classification Accuracy: {accuracy:.2%} (Correctly predicts Over/Under {CORNER_LINE} corners)")

#     # --- 3. Save Models and Feature List ---
#     print("\nSaving models and feature list...")
#     joblib.dump(regr_model, 'corner_regression_model.joblib')
#     joblib.dump(class_model, 'corner_classification_model.joblib')
#     joblib.dump(features, 'model_features.joblib')
    
#     print("✅ Models and feature list saved successfully.")
#     print("\n--- Script Finished ---")

# if __name__ == '__main__':
#     input_csv = 'top_5_leagues_cleaned_data.csv'
#     train_model(input_csv)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def feature_engineering(df):
    """
    This is the simpler, more effective feature engineering function that produced
    the best model results. It calculates rolling averages, strength ratings, and H2H stats.
    """
    print("Starting feature engineering...")
    df = df.sort_values('MatchDate').copy()

    match_list = []
    for index, row in df.iterrows():
        match_list.append({
            'LeagueName': row['LeagueName'], 'team': row['HomeTeam'], 'opponent': row['AwayTeam'], 'date': row['MatchDate'],
            'goals_for': row['FTHome'], 'goals_against': row['FTAway'], 'shots_for': row['HomeShots'],
            'shots_against': row['AwayShots'], 'shots_target_for': row['HomeTarget'], 'corners_for': row['HomeCorners'], 'corners_against': row['AwayCorners'],
        })
        match_list.append({
            'LeagueName': row['LeagueName'], 'team': row['AwayTeam'], 'opponent': row['HomeTeam'], 'date': row['MatchDate'],
            'goals_for': row['FTAway'], 'goals_against': row['FTHome'], 'shots_for': row['AwayShots'],
            'shots_against': row['HomeShots'], 'shots_target_for': row['AwayTarget'], 'corners_for': row['AwayCorners'], 'corners_against': row['HomeCorners'],
        })
        
    team_stats_df = pd.DataFrame(match_list).sort_values(by=['team', 'date']).reset_index(drop=True)

    stats_to_average = ['goals_for', 'goals_against', 'shots_for', 'shots_against', 'shots_target_for', 'corners_for', 'corners_against']
    
    for stat in stats_to_average:
        rolling_avg = team_stats_df.groupby('team')[stat].transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
        team_stats_df[f'avg_{stat}_L5'] = rolling_avg

    df = df.merge(team_stats_df, left_on=['HomeTeam', 'MatchDate'], right_on=['team', 'date'], how='left').rename(columns={f'avg_{s}_L5': f'home_{s}_L5' for s in stats_to_average})
    df = df.merge(team_stats_df, left_on=['AwayTeam', 'MatchDate'], right_on=['team', 'date'], how='left', suffixes=('_home', '_away')).rename(columns={f'avg_{s}_L5': f'away_{s}_L5' for s in stats_to_average})

    df.rename(columns={'LeagueName_home': 'LeagueName'}, inplace=True)
    cols_to_drop = [c for c in df.columns if '_home' in c or '_away' in c and c not in ['HomeTeam', 'AwayTeam']]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    league_avg_stats = df.groupby('LeagueName').agg(league_avg_corners=('HomeCorners', lambda x: (x.sum() + df.loc[x.index, 'AwayCorners'].sum()) / (2 * len(x)))).reset_index()
    df = df.merge(league_avg_stats, on='LeagueName', how='left')
    
    df['home_attack_strength'] = df['home_corners_for_L5'] / df['league_avg_corners']
    df['home_defense_strength'] = df['home_corners_against_L5'] / df['league_avg_corners']
    df['away_attack_strength'] = df['away_corners_for_L5'] / df['league_avg_corners']
    df['away_defense_strength'] = df['away_corners_against_L5'] / df['league_avg_corners']

    h2h_corners = []
    df['total_corners_temp'] = df['HomeCorners'] + df['AwayCorners']
    for index, row in df.iterrows():
        past_matches = df[(df['MatchDate'] < row['MatchDate']) & (((df['HomeTeam'] == row['HomeTeam']) & (df['AwayTeam'] == row['AwayTeam'])) | ((df['HomeTeam'] == row['AwayTeam']) & (df['AwayTeam'] == row['HomeTeam'])))].tail(3)
        h2h_corners.append(past_matches['total_corners_temp'].mean() if not past_matches.empty else np.nan)
    df['h2h_avg_corners_L3'] = h2h_corners
    df['h2h_avg_corners_L3'].fillna(df['league_avg_corners'], inplace=True)
    df.drop(columns=['total_corners_temp'], inplace=True)

    df.dropna(inplace=True)
    print(f"✅ Feature engineering complete. {len(df)} matches available for training.")
    return df

def train_multi_line_model(input_filepath):
    print(f"Loading data from {input_filepath}...")
    df = pd.read_csv(input_filepath, parse_dates=['MatchDate'])
    
    if 'LeagueName' not in df.columns and 'Division' in df.columns:
        leagues_map = {'E0': 'Premier League', 'SP1': 'La Liga', 'I1': 'Serie A', 'D1': 'Bundesliga', 'F1': 'Ligue 1'}
        df['LeagueName'] = df['Division'].map(leagues_map)
        df.dropna(subset=['LeagueName'], inplace=True)

    df_featured = feature_engineering(df)
    df_featured['total_corners'] = df_featured['HomeCorners'] + df_featured['AwayCorners']

    features = [
        'home_corners_for_L5', 'home_corners_against_L5', 'away_corners_for_L5', 'away_corners_against_L5',
        'home_shots_for_L5', 'away_shots_for_L5', 'home_attack_strength', 'home_defense_strength',
        'away_attack_strength', 'away_defense_strength', 'h2h_avg_corners_L3',
        'OddHome', 'OddDraw', 'OddAway'
    ]
    
    X = df_featured[features]
    
    # --- 1. Train Regression Model (Stays the same) ---
    print("\n--- Training Regression Model ---")
    y_regr = df_featured['total_corners']
    X_train, X_test, y_regr_train, y_regr_test = train_test_split(X, y_regr, test_size=0.2, random_state=42)
    regr_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42, n_jobs=-1)
    regr_model.fit(X_train, y_regr_train, verbose=False)
    mae = mean_absolute_error(y_regr_test, regr_model.predict(X_test))
    print(f"📈 Regression MAE: {mae:.4f}")
    joblib.dump(regr_model, 'corner_regression_model.joblib')
    joblib.dump(features, 'model_features.joblib')
    print("✅ Regression model saved.")

    # --- 2. Train Multiple Classification Models ---
    corner_lines = [8.5, 9.5, 10.5, 11.5, 12.5]
    for line in corner_lines:
        print(f"\n--- Training Classifier for Over/Under {line} ---")
        target_col = f'over_{str(line).replace(".", "_")}'
        df_featured[target_col] = (df_featured['total_corners'] > line).astype(int)
        y_class = df_featured[target_col]
        
        X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
        
        class_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=500, learning_rate=0.05, max_depth=3, scale_pos_weight=1, use_label_encoder=False, random_state=42, n_jobs=-1)
        class_model.fit(X_train, y_class_train, verbose=False)
        
        accuracy = accuracy_score(y_class_test, class_model.predict(X_test))
        print(f"🎯 Accuracy for O/U {line}: {accuracy:.2%}")
        
        model_filename = f'corner_classifier_o{str(line).replace(".", "")}.joblib'
        joblib.dump(class_model, model_filename)
        print(f"✅ Classifier for O/U {line} saved to {model_filename}")

    print("\n--- All models trained and saved successfully ---")

if __name__ == '__main__':
    input_csv = 'top_5_leagues_cleaned_data.csv'
    train_multi_line_model(input_csv)