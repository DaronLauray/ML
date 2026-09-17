from pathlib import Path
from flask import Flask, render_template, request, jsonify

try:
    from .nba_winner_predictor import make_prediction, load_model
except ImportError:
    from nba_winner_predictor import make_prediction, load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__, template_folder=str(PROJECT_ROOT / 'templates'))

@app.route('/')
def home():
    try:
        model, features, team_features = load_model()
        teams = sorted(team_features['teamName'].unique().tolist())
        return render_template('index.html', teams=teams)
    except Exception as e:
        return f"Error loading model: {e}", 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        home_team = data.get('home_team', '').strip()
        away_team = data.get('away_team', '').strip()
        date = data.get('date', None)
        
        if not home_team or not away_team:
            return jsonify({'error': 'Both teams required'}), 400
        
        result = make_prediction(home_team, away_team, date)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/teams', methods=['GET'])
def get_teams():
    try:
        model, features, team_features = load_model()
        teams = sorted(team_features['teamName'].dropna().astype(str).unique().tolist())
        return jsonify({'teams': teams})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
