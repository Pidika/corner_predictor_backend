import os
import requests
from dotenv import load_dotenv
from team_name_mapper import normalize_team_name
import json

load_dotenv()

ODDS_API_KEY = os.getenv('ODDS_API_KEY')

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
    if not ODDS_API_KEY:
        print("Error: ODDS_API_KEY not found. Cannot fetch odds.")
        return {"error": "Odds API key is not configured on the server."}

    upcoming_fixtures = []
    
    for league_name, league_key in LEAGUE_KEYS.items():
        print(f"Fetching odds for {league_name}...")
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds/"
            params = {
                'apiKey': ODDS_API_KEY, 'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            
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
                    "home_team": normalized_home_team, "away_team": normalized_away_team,
                    "league_name": league_name, "odd_home": home_odd,
                    "odd_draw": draw_odd, "odd_away": away_odd,
                })
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not fetch odds for {league_name}. Error: {e}")
            continue

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(upcoming_fixtures, f, indent=4)
        
    print(f"\n✅ Successfully fetched and saved a total of {len(upcoming_fixtures)} matches to {output_json_path}.")
    return {"status": "success", "matches_fetched": len(upcoming_fixtures)}

if __name__ == "__main__":
    fetch_and_save_upcoming_matches()
