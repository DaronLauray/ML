import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(PROJECT_ROOT / 'Data')
MODEL_PATH = str(Path(__file__).resolve().parent / 'nba_winner_model.joblib')

TEAM_STATS_FILE = 'TeamStatistics.csv'
TEAM_ADV_FILE = 'TeamStatisticsAdvanced.csv'
PLAYER_USAGE_FILE = 'PlayerStatisticsUsage.csv'
PLAYER_STATS_FILE = 'PlayerStatistics.csv'
GAMES_FILE = 'Games.csv'

FEATURE_COLUMNS = [
    'avg5_teamScore', 'avg5_assists', 'avg5_turnovers', 'avg5_reboundsTotal',
    'avg5_fieldGoalsPercentage', 'avg5_threePointersPercentage', 'avg5_freeThrowsPercentage',
    'avg5_pointsFastBreak', 'avg5_pointsFromTurnovers', 'avg5_pointsInThePaint',
    'avg5_pointsSecondChance', 'avg5_plusMinusPoints', 'avg5_netRating',
    'avg5_offRating', 'avg5_defRating', 'avg5_top3_points', 'avg5_max_usgPct'
]

def load_games():
    path = os.path.join(DATA_DIR, GAMES_FILE)
    nrows = int(os.environ.get('SAMPLE_N')) if os.environ.get('SAMPLE_N') else None
    games = pd.read_csv(path, low_memory=False, dtype={'gameId': str, 'hometeamId': str, 'awayteamId': str, 'winner': str}, nrows=nrows)
    games['gameDateTimeEst'] = pd.to_datetime(games['gameDateTimeEst'], utc=True, errors='coerce')
    games = games.dropna(subset=['gameDateTimeEst'])
    games = games.sort_values('gameDateTimeEst').reset_index(drop=True)
    games['home_win'] = (games['winner'] == games['hometeamId']).astype(int)
    games['gameId'] = games['gameId'].astype(str)
    return games


def load_team_stats():
    path = os.path.join(DATA_DIR, TEAM_STATS_FILE)
    usecols = [
        'gameId', 'gameDateTimeEst', 'teamCity', 'teamName', 'teamId',
        'home', 'win', 'teamScore', 'assists', 'turnovers', 'reboundsTotal',
        'fieldGoalsPercentage', 'threePointersPercentage', 'freeThrowsPercentage',
        'plusMinusPoints', 'pointsFastBreak', 'pointsFromTurnovers',
        'pointsInThePaint', 'pointsSecondChance', 'seasonWins', 'seasonLosses'
    ]
    nrows = int(os.environ.get('SAMPLE_N')) if os.environ.get('SAMPLE_N') else None
    df = pd.read_csv(path, low_memory=False, usecols=usecols, dtype={'gameId': str, 'teamId': str}, nrows=nrows)
    df['gameDateTimeEst'] = pd.to_datetime(df['gameDateTimeEst'], utc=True, errors='coerce')
    df['gameId'] = df['gameId'].astype(str)
    df['teamId'] = df['teamId'].astype(str)
    numeric_cols = [
        'teamScore', 'opponentScore', 'assists', 'blocks', 'steals',
        'fieldGoalsPercentage', 'threePointersPercentage', 'freeThrowsPercentage',
        'reboundsTotal', 'turnovers', 'plusMinusPoints', 'pointsFastBreak',
        'pointsFromTurnovers', 'pointsInThePaint', 'pointsSecondChance',
        'seasonWins', 'seasonLosses'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_team_advanced():
    path = os.path.join(DATA_DIR, TEAM_ADV_FILE)
    nrows = int(os.environ.get('SAMPLE_N')) if os.environ.get('SAMPLE_N') else None
    df = pd.read_csv(path, low_memory=False, dtype={'gameId': str, 'teamId': str}, nrows=nrows)
    df['gameDateTimeEst'] = pd.to_datetime(df['gameDateTimeEst'], utc=True, errors='coerce')
    df['gameId'] = df['gameId'].astype(str)
    df['teamId'] = df['teamId'].astype(str)
    for col in ['netRating', 'offRating', 'defRating']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_player_usage():
    path = os.path.join(DATA_DIR, PLAYER_USAGE_FILE)
    usecols = ['gameId', 'playerteamName', 'usgPct']
    nrows = int(os.environ.get('SAMPLE_N')) if os.environ.get('SAMPLE_N') else None
    df = pd.read_csv(path, low_memory=False, usecols=usecols, dtype={'gameId': str, 'playerteamName': str}, nrows=nrows)
    df['usgPct'] = pd.to_numeric(df['usgPct'], errors='coerce')
    df['gameId'] = df['gameId'].astype(str)
    return df


def load_player_stats():
    path = os.path.join(DATA_DIR, PLAYER_STATS_FILE)
    usecols = ['gameId', 'playerteamName', 'points']
    # use python engine and skip problematic lines to avoid tokenization OOM on malformed rows
    nrows = int(os.environ.get('SAMPLE_N')) if os.environ.get('SAMPLE_N') else None
    df = pd.read_csv(
        path,
        low_memory=True,
        usecols=usecols,
        dtype={'gameId': str, 'playerteamName': str},
        engine='python',
        on_bad_lines='skip',
        nrows=nrows,
    )
    df['points'] = pd.to_numeric(df['points'], errors='coerce')
    df['gameId'] = df['gameId'].astype(str)
    return df


def build_player_features():
    usage = load_player_usage()
    stats = load_player_stats()
    top_usage = (
        usage.groupby(['gameId', 'playerteamName'])['usgPct']
        .max()
        .reset_index(name='max_usgPct')
    )
    stats = stats.sort_values(['gameId', 'playerteamName', 'points'], ascending=[True, True, False])
    top3_points = (
        stats.groupby(['gameId', 'playerteamName'])
        .head(3)
        .groupby(['gameId', 'playerteamName'])['points']
        .sum()
        .reset_index(name='top3_points')
    )
    player_features = pd.merge(top_usage, top3_points, on=['gameId', 'playerteamName'], how='outer')
    player_features['playerteamName'] = player_features['playerteamName'].astype(str)
    player_features['max_usgPct'] = player_features['max_usgPct'].fillna(0)
    player_features['top3_points'] = player_features['top3_points'].fillna(0)
    return player_features


def build_team_game_features():
    team_stats = load_team_stats()
    advanced = load_team_advanced()
    player_features = build_player_features()

    base = pd.merge(
        team_stats,
        advanced[['gameId', 'teamId', 'netRating', 'offRating', 'defRating']],
        on=['gameId', 'teamId'],
        how='left',
    )

    base = pd.merge(
        base,
        player_features,
        left_on=['gameId', 'teamName'],
        right_on=['gameId', 'playerteamName'],
        how='left',
    )
    base['top3_points'] = base['top3_points'].fillna(0)
    base['max_usgPct'] = base['max_usgPct'].fillna(0)

    base = base.sort_values(['teamId', 'gameDateTimeEst'])
    shift_cols = [
        'teamScore', 'assists', 'turnovers', 'reboundsTotal', 'fieldGoalsPercentage',
        'threePointersPercentage', 'freeThrowsPercentage', 'pointsFastBreak',
        'pointsFromTurnovers', 'pointsInThePaint', 'pointsSecondChance',
        'plusMinusPoints', 'netRating', 'offRating', 'defRating',
        'top3_points', 'max_usgPct'
    ]
    group = base.groupby('teamId', group_keys=False)
    for col in shift_cols:
        if col in base.columns:
            base[f'lag1_{col}'] = group[col].shift(1)
            base[f'avg5_{col}'] = group[col].shift(1).rolling(5, min_periods=1).mean()

    features = base[
        ['gameId', 'teamId', 'teamCity', 'teamName', 'home', 'win', 'gameDateTimeEst'] +
        [c for c in base.columns if c.startswith('avg5_')]
    ].copy()
    features = features.dropna(subset=['avg5_teamScore'])
    # Ensure gameDateTimeEst is datetime before returning to avoid joblib serialization issues
    features['gameDateTimeEst'] = pd.to_datetime(features['gameDateTimeEst'], utc=True, errors='coerce')
    features = features.dropna(subset=['gameDateTimeEst'])
    return features


def build_matchup_dataset():
    games = load_games()
    team_features = build_team_game_features()

    home_features = team_features[team_features['home'] == 1].copy()
    away_features = team_features[team_features['home'] == 0].copy()
    home_features = home_features.rename(columns={
        col: f'home_{col}' for col in home_features.columns
        if col not in ['gameId', 'teamId', 'gameDateTimeEst', 'home', 'win', 'teamCity', 'teamName']
    })
    away_features = away_features.rename(columns={
        col: f'away_{col}' for col in away_features.columns
        if col not in ['gameId', 'teamId', 'gameDateTimeEst', 'home', 'win', 'teamCity', 'teamName']
    })

    merged = pd.merge(
        games,
        home_features,
        left_on=['gameId'],
        right_on=['gameId'],
        how='inner',
    )
    merged = pd.merge(
        merged,
        away_features,
        left_on=['gameId'],
        right_on=['gameId'],
        how='inner',
        suffixes=('_home', '_away')
    )

    feature_cols = []
    for stat in FEATURE_COLUMNS:
        if f'home_{stat}' in merged.columns and f'away_{stat}' in merged.columns:
            merged[f'diff_{stat}'] = merged[f'home_{stat}'] - merged[f'away_{stat}']
            feature_cols.append(f'diff_{stat}')
    if not feature_cols:
        raise RuntimeError('No matchup feature columns were created. Check data availability and feature names.')
    merged['home_rate'] = merged['hometeamId']
    merged = merged.dropna(subset=feature_cols)

    return merged, feature_cols


def train_model():
    data, feature_cols = build_matchup_dataset()
    data = data.sort_values('gameDateTimeEst')
    X = data[feature_cols]
    y = data['home_win']
    if len(data) < 50:
        raise RuntimeError('Not enough matchup examples to train reliably.')

    split_index = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    joblib.dump({
        'model': model,
        'features': feature_cols,
        'team_features': build_team_game_features(),
    }, MODEL_PATH)

    print('Training complete')
    print('Training examples:', len(X_train), 'Test examples:', len(X_test))
    print('Test accuracy:', score)
    print(report)
    print('Saved model to', MODEL_PATH)
    return model, feature_cols, data


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError('Model file not found. Run training first.')
    payload = joblib.load(MODEL_PATH)
    team_features = payload['team_features'].copy()
    # Ensure gameDateTimeEst is datetime after loading from joblib
    if 'gameDateTimeEst' in team_features.columns:
        team_features['gameDateTimeEst'] = pd.to_datetime(team_features['gameDateTimeEst'], utc=True, errors='coerce')
        team_features = team_features.dropna(subset=['gameDateTimeEst'])
    if 'teamName' in team_features.columns:
        team_features = team_features.dropna(subset=['teamName']).copy()
        team_features['teamName'] = team_features['teamName'].astype(str).str.strip()
        team_features = team_features[team_features['teamName'].ne('')]
    return payload['model'], payload['features'], team_features


def make_prediction(home_team, away_team, date=None):
    model, feature_cols, team_features = load_model()
    home_team = str(home_team).strip()
    away_team = str(away_team).strip()

    teams = team_features['teamName'].unique().tolist()
    if home_team not in teams or away_team not in teams:
        raise ValueError('Team names not found in dataset. Available teams: ' + ', '.join(sorted(teams)))

    if date is None:
        date = team_features['gameDateTimeEst'].max()
    else:
        date = pd.to_datetime(date, utc=True, errors='coerce')
        if pd.isna(date):
            raise ValueError('Invalid date format')

    def last_team_stats(name):
        subset = team_features[(team_features['teamName'] == name) & (team_features['gameDateTimeEst'] < date)]
        if subset.empty:
            return None
        row = subset.iloc[-1]
        return row

    home_row = last_team_stats(home_team)
    away_row = last_team_stats(away_team)
    if home_row is None or away_row is None:
        raise ValueError('Not enough historical games for one or both teams before the given date.')

    features = {}
    for stat in FEATURE_COLUMNS:
        features[f'diff_{stat}'] = home_row[stat] - away_row[stat]
    X = pd.DataFrame([features]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features = X.iloc[0].to_dict()
    proba = model.predict_proba(X)[0]
    home_prob = float(proba[1])
    away_prob = float(proba[0])
    predicted = home_team if home_prob >= away_prob else away_team
    importance = model.feature_importances_
    sorted_importance = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    explanation = [f'{name}: {weight:.4f}' for name, weight in sorted_importance[:6]]
    return {
        'prediction': predicted,
        'home_probability': home_prob,
        'away_probability': away_prob,
        'explanation': explanation,
        'home_team': home_team,
        'away_team': away_team,
        'feature_values': features,
    }


def run_console_ui():
    print('NBA Winner Predictor')
    print('Training model from historical team and player data...')
    if not os.path.exists(MODEL_PATH):
        train_model()
    while True:
        home = input('Enter home team name: ').strip()
        away = input('Enter away team name: ').strip()
        date = input('Enter game date (YYYY-MM-DD) or leave blank for latest: ').strip() or None
        try:
            result = make_prediction(home, away, date)
            print(f"Predicted winner: {result['prediction']}")
            print(f"Home win probability: {result['home_probability']:.2%}")
            print(f"Away win probability: {result['away_probability']:.2%}")
            print('Top contributing factors:')
            print('\n'.join(result['explanation']))
        except Exception as exc:
            print('Error:', exc)
        cont = input('Predict another game? [y/N]: ').strip().lower()
        if cont != 'y':
            break


def main():
    print('NBA Winner Predictor')
    if not os.path.exists(MODEL_PATH):
        train_model()
    run_console_ui()

if __name__ == '__main__':
    main()
