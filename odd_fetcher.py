import os
import requests
from dotenv import load_dotenv
from team_name_mapper import normalize_team_name
import json

load_dotenv()

# Ensure you have your key in the .env file as ODDS_API_KEY
ODDS_API_KEY = os.getenv('ODDS_API_KEY')
if not ODDS_API_KEY:
    print("Error: THE_ODDS_API_KEY not found in .env file. Cannot fetch odds.")
    exit()

# The Odds API uses specific keys for each league
LEAGUE_KEYS = {
    'Premier League': 'soccer_epl',
    'La Liga': 'soccer_spain_la_liga',
    'Serie A': 'soccer_italy_serie_a',
    'Bundesliga': 'soccer_germany_bundesliga',
    'Ligue 1': 'soccer_france_ligue_one' 
}

def fetch_and_save_upcoming_matches(output_json_path='public/upcoming_matches.json'):
    """
    Fetches upcoming matches for all leagues, normalizes team names,
    and saves the consolidated list to a single JSON file.
    """
    upcoming_fixtures = []
    
    for league_name, league_key in LEAGUE_KEYS.items():
        print(f"Fetching odds for {league_name}...")
        
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds/"
            params = {
                'apiKey': ODDS_API_KEY,
                'regions': 'eu',
                'markets': 'h2h',
                'oddsFormat': 'decimal'
            }
            response = requests.get(url, params=params)
            response.raise_for_status() # Will raise an error for bad status codes
            
            data = response.json()
            print(f"  -> Found {len(data)} upcoming matches.")

            for match in data:
                bookmaker = match.get('bookmakers', [])[0] if match.get('bookmakers') else None
                if not bookmaker: continue

                market = next((m for m in bookmaker.get('markets', []) if m['key'] == 'h2h'), None)
                if not market: continue
                
                home_odd = next((o['price'] for o in market['outcomes'] if o['name'] == match['home_team']), None)
                away_odd = next((o['price'] for o in market['outcomes'] if o['name'] == match['away_team']), None)
                draw_odd = next((o['price'] for o in market['outcomes'] if o['name'] == 'Draw'), None)

                if not all([home_odd, away_odd, draw_odd]): continue

                normalized_home_team = normalize_team_name(match['home_team'])
                normalized_away_team = normalize_team_name(match['away_team'])
                
                upcoming_fixtures.append({
                    "home_team": normalized_home_team,
                    "away_team": normalized_away_team,
                    "league_name": league_name,
                    "odd_home": home_odd,
                    "odd_draw": draw_odd,
                    "odd_away": away_odd,
                })
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not fetch odds for {league_name}. Error: {e}")
            continue

    # Save the final consolidated list to the JSON file
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(upcoming_fixtures, f, indent=4)
        
    print(f"\n✅ Successfully fetched and saved a total of {len(upcoming_fixtures)} matches to {output_json_path}.")

if __name__ == "__main__":
    fetch_and_save_upcoming_matches()

    
# my own odd fetcher which is working well
# import os
# import requests
# from dotenv import load_dotenv
# from team_name_mapper import normalize_team_name
# import json

# load_dotenv()

# ODDS_API_KEY = os.getenv('ODDS_API_KEY')
# if not ODDS_API_KEY:
#     print("Warning: THE_ODDS_API_KEY not found in .env file. Odds fetching will not work.")

# # The Odds API uses specific keys for each league
# LEAGUE_KEYS = {
#     'Premier League': 'soccer_epl',
#     'La Liga': 'soccer_spain_la_liga',
#     'Serie A': 'soccer_italy_serie_a',
#     'Bundesliga': 'soccer_germany_bundesliga',
#     'Ligue 1': 'soccer_france_ligue_one'
# }

# def fetch_upcoming_matches(output_json_path='public/upcoming_features.json'):
   
#     if not ODDS_API_KEY:
#         return {"error": "Odds API key is not configured on the server."}

#     upcoming_fixtures = []
    
#     # --- FIX: Loop through each league and make an individual API call ---
#     for league_name, league_key in LEAGUE_KEYS.items():
#         print(f"Fetching odds for {league_name}...")
        
#         try:
#             url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds/"
#             params = {
#                 'apiKey': ODDS_API_KEY,
#                 'regions': 'eu', # uk, us, eu, au. Use 'eu' for better bookmaker coverage.
#                 'markets': 'h2h', # Head-to-head (1x2) odds
#                 'oddsFormat': 'decimal'
#             }
#             response = requests.get(url, params=params)
#             response.raise_for_status()
#             print(f"response from the api {response.json}")
#             data = response.json()
#             print(f"  -> Found {len(data)} upcoming matches.")

#             for match in data:
#                 bookmaker = match.get('bookmakers', [])[0] if match.get('bookmakers') else None
#                 if not bookmaker:
#                     continue

#                 market = next((m for m in bookmaker.get('markets', []) if m['key'] == 'h2h'), None)
#                 if not market:
#                     continue
                
#                 home_odd = next((o['price'] for o in market['outcomes'] if o['name'] == match['home_team']), None)
#                 away_odd = next((o['price'] for o in market['outcomes'] if o['name'] == match['away_team']), None)
#                 draw_odd = next((o['price'] for o in market['outcomes'] if o['name'] == 'Draw'), None)

#                 if not all([home_odd, away_odd, draw_odd]):
#                     continue
#                 normalized_home_team = normalize_team_name(match['home_team'])
#                 normalized_away_team = normalize_team_name(match['away_team'])
#                 upcoming_fixtures.append({
#                     "home_team": normalized_home_team,
#                     "away_team": normalized_away_team,
#                     "league_name": league_name, # Use the name from our loop
#                     "odd_home": home_odd,
#                     "odd_draw": draw_odd,
#                     "odd_away": away_odd,
#                 })
#                 os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
#                 with open(output_json_path, 'w') as f:
#                  json.dump(upcoming_fixtures, f, indent=4)
#                 print(f"✅ features saved to  {output_json_path}")
#         except requests.exceptions.RequestException as e:
#             # If one league fails, print a warning but continue to the next
#             print(f"Warning: Could not fetch odds for {league_name}. Error: {e}")
#             continue

#     print(f"\nSuccessfully fetched a total of {len(upcoming_fixtures)} matches across all leagues.")
#     return upcoming_fixtures
# if __name__ == "__main__":
#   fetch_upcoming_matches()


# import os
# import requests
# from dotenv import load_dotenv
# import json
# import time

# load_dotenv()

# ODDS_API_KEY = os.getenv('ODDS_API_KEY')
# if not ODDS_API_KEY:
#     print("Warning: THE_ODDS_API_KEY not found in .env file. Odds fetching will not work.")

# # The Odds API uses specific keys for each league
# LEAGUE_KEYS = {
#     'Premier League': 'soccer_epl',
#     'La Liga': 'soccer_spain_la_liga',
#     'Serie A': 'soccer_italy_serie_a',
#     'Bundesliga': 'soccer_germany_bundesliga',
#     'Ligue 1': 'soccer_france_ligue_one'
# }

# def fetch_upcoming_matches_and_log_responses():
#     """
#     Fetches upcoming football matches and logs the raw JSON responses
#     from The Odds API for both H2H and alternate_totals_corners markets.
#     This function is for debugging to inspect available corner options.
#     """
#     if not ODDS_API_KEY:
#         print({"error": "Odds API key is not configured on the server."})
#         return

#     # --- Step 1: Fetch H2H odds and event IDs for all matches and log raw response ---
#     for league_name, league_key in LEAGUE_KEYS.items():
#         print(f"\n--- Fetching H2H odds for {league_name} (Logging Raw Response) ---")
        
#         try:
#             url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds/"
#             params = {
#                 'apiKey': ODDS_API_KEY,
#                 'regions': 'eu', # uk, us, eu, au. Use 'eu' for better bookmaker coverage.
#                 'markets': 'h2h', # Only request h2h here
#                 'oddsFormat': 'decimal'
#             }
#             response = requests.get(url, params=params)
#             response.raise_for_status() 
            
#             # Print the full JSON response from the API for H2H call
#             print(json.dumps(response.json(), indent=2)) 
            
#             # We don't need to process the data beyond logging for this request,
#             # but we still need event IDs for the second step.
#             data = response.json()
            
#             # Store necessary info for the second step
#             matches_for_corners = []
#             for match in data:
#                 matches_for_corners.append({
#                     "id": match['id'], 
#                     "home_team": match['home_team'],
#                     "away_team": match['away_team'],
#                     "sport_key": league_key 
#                 })

#             print(f"--- Finished logging H2H response for {league_name}. Found {len(matches_for_corners)} matches for corner checks. ---\n")

#             # --- Step 2: Fetch alternate_totals_corners for each individual match and log raw response ---
#             if matches_for_corners:
#                 print(f"\n--- Fetching Alternate Total Corners for {league_name} (Logging Raw Response per match) ---")
#                 for fixture_info in matches_for_corners:
#                     event_id = fixture_info['id']
#                     sport_key = fixture_info['sport_key']
                    
#                     print(f"\n--- Corner Odds for {fixture_info['home_team']} vs {fixture_info['away_team']} (ID: {event_id}) ---")

#                     try:
#                         url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds/"
#                         params = {
#                             'apiKey': ODDS_API_KEY,
#                             'regions': 'eu',
#                             'markets': 'alternate_totals_corners', 
#                             'oddsFormat': 'decimal'
#                         }
#                         response = requests.get(url, params=params)
#                         response.raise_for_status()

#                         # Print the full JSON response for the individual event call
#                         print(json.dumps(response.json(), indent=2))
                        
#                         # Implement a small delay to respect API rate limits
#                         time.sleep(0.5) 

#                     except requests.exceptions.RequestException as e:
#                         print(f"    -> ERROR: Could not fetch corner odds for {fixture_info['home_team']} vs {fixture_info['away_team']}. Error: {e}")
#                     except Exception as e:
#                         print(f"    -> ERROR: An unexpected error occurred during corner odds fetch for {fixture_info['home_team']} vs {fixture_info['away_team']}: {e}")
#                     print(f"--- End of Corner Odds for {fixture_info['home_team']} vs {fixture_info['away_team']} ---\n")
#                 print(f"--- Finished logging Alternate Total Corners for {league_name}. ---\n")
#             else:
#                 print(f"No matches found with H2H odds for {league_name} to check corner odds.")

#         except requests.exceptions.RequestException as e:
#             print(f"Warning: Could not fetch initial H2H data for {league_name}. Error: {e}")
#         except Exception as e:
#             print(f"An unexpected error occurred for {league_name}: {e}")

# # Example usage (for testing purposes)
# if __name__ == "__main__":
#     # Ensure you have a .env file in the same directory with ODDS_API_KEY="your_api_key_here"
    
#     # This will print raw JSON responses to your console.
#     # Be aware of API credit usage and console output volume.
#     fetch_upcoming_matches_and_log_responses()