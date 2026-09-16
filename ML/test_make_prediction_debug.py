import traceback
from nba_winner_predictor import load_model, make_prediction
import pandas as pd

print('Loading model and team features...')
model, features, team_features = load_model()
print('\nteam_features dtypes:')
print(team_features.dtypes)

col = 'gameDateTimeEst'
if col in team_features.columns:
    vals = team_features[col]
    # show sample types
    type_counts = vals.apply(lambda x: type(x)).value_counts()
    print('\nSample value types for gameDateTimeEst:')
    print(type_counts)
    # show problematic rows only when the column did not normalize to datetime
    if pd.api.types.is_datetime64_any_dtype(team_features[col]):
        bad = team_features.iloc[0:0]
    else:
        bad = team_features.copy()
    print(f'\nRows where dtype is not datetime (count={len(bad)}):')
    if len(bad) > 0:
        print(bad.head(10).to_dict(orient='records'))
else:
    print('gameDateTimeEst column not found')

# Try a prediction to reproduce the error
teams = sorted(team_features['teamName'].unique().tolist())
print('\nAvailable teams sample:', teams[:5])

home = teams[0]
away = teams[1] if len(teams) > 1 else teams[0]
print(f'Testing make_prediction with home={home}, away={away}')
try:
    res = make_prediction(home, away)
    print('Prediction result:', res)
except Exception as e:
    print('Exception occurred during make_prediction:')
    traceback.print_exc()
