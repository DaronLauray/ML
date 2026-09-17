from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'ML' / 'nba_winner_model.joblib'
EXPORT_PATH = ROOT / 'model_export.json'
PROFILES_PATH = ROOT / 'team_profiles.json'


def export_model() -> None:
    payload = joblib.load(MODEL_PATH)
    model = payload['model']
    feature_names = payload['features']

    model_export = {
        'feature_names': feature_names,
        'n_estimators': len(model.estimators_),
        'n_classes': [int(v) for v in model.classes_.tolist()],
        'trees': [],
    }

    for estimator in model.estimators_:
        tree = estimator.tree_
        nodes = []
        for idx in range(tree.node_count):
            left = int(tree.children_left[idx])
            right = int(tree.children_right[idx])
            is_leaf = left == right
            node = {
                'node_id': int(idx),
                'is_leaf': is_leaf,
                'value': [float(v) for v in tree.value[idx][0].tolist()],
            }
            if not is_leaf:
                node['feature_index'] = int(tree.feature[idx])
                node['threshold'] = float(tree.threshold[idx])
                node['left'] = left
                node['right'] = right
            nodes.append(node)
        model_export['trees'].append(nodes)

    EXPORT_PATH.write_text(json.dumps(model_export, separators=(',', ':')))


def export_profiles() -> None:
    payload = joblib.load(MODEL_PATH)
    team_features = payload['team_features'].copy()
    latest = team_features.sort_values(['teamName', 'gameDateTimeEst']).drop_duplicates('teamName', keep='last')
    latest = latest.dropna(subset=['teamName']).copy()
    latest['teamName'] = latest['teamName'].astype(str).str.strip()
    latest = latest[latest['teamName'] != '']

    profiles: dict[str, dict[str, float]] = {}
    for row in latest.itertuples(index=False):
        name = str(row.teamName).strip()
        profile: dict[str, float] = {}
        for feature_name in [
            'avg5_teamScore', 'avg5_assists', 'avg5_turnovers', 'avg5_reboundsTotal',
            'avg5_fieldGoalsPercentage', 'avg5_threePointersPercentage', 'avg5_freeThrowsPercentage',
            'avg5_pointsFastBreak', 'avg5_pointsFromTurnovers', 'avg5_pointsInThePaint',
            'avg5_pointsSecondChance', 'avg5_plusMinusPoints', 'avg5_netRating', 'avg5_offRating',
            'avg5_defRating', 'avg5_top3_points', 'avg5_max_usgPct'
        ]:
            value = getattr(row, feature_name, 0)
            profile[feature_name] = 0.0 if pd.isna(value) else float(value)
        profiles[name] = profile

    PROFILES_PATH.write_text(json.dumps(profiles, separators=(',', ':')))


if __name__ == '__main__':
    export_model()
    export_profiles()
    print(f'Wrote {EXPORT_PATH.name} ({EXPORT_PATH.stat().st_size} bytes)')
    print(f'Wrote {PROFILES_PATH.name} ({PROFILES_PATH.stat().st_size} bytes)')
