# NBA Winner Predictor

An AI-powered system to predict NBA game winners based on historical team and player statistics.

## Overview

This project uses a **RandomForestClassifier** trained on NBA team statistics, advanced metrics, and player performance data to predict game outcomes. The system achieves ~58% test accuracy and identifies key statistical factors that influence home team win probability.

## Features

- **Predictive Model**: Trained on 20+ seasons of NBA data
- **Feature Engineering**: 
  - Lag-1 and rolling 5-game averages of team stats
  - Player-level aggregation (top-3 scorers, max usage percentage)
  - Home/Away differential features
- **Web UI**: Flask-based interface for interactive predictions
- **Interpretability**: Feature importance rankings to explain predictions
- **Statistical Factors**: Identifies which metrics most influence game outcomes

## Data Sources

- `Games.csv` - Game metadata and winners
- `TeamStatistics.csv` - Team scoring, rebounds, assists, turnovers, etc.
- `TeamStatisticsAdvanced.csv` - Advanced metrics (net rating, pace, eFG%, etc.)
- `PlayerStatistics.csv` - Player scoring data
- `PlayerStatisticsUsage.csv` - Player usage percentages

## Installation

### 1. Create Virtual Environment

```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn joblib flask matplotlib
```

### 3. Verify Data

Ensure the `Data/` folder contains all required CSV files from the data sources listed above.

## Usage

### Option A: Web Interface (Recommended)

```bash
python app.py
```

Then open http://localhost:5000 in your browser. Select home and away teams to get predictions with probability and top factors.

### Option B: Console Interface

```bash
python nba_winner_predictor.py
```

This launches an interactive console where you enter team names and dates.

### Option C: Programmatic API

```python
from nba_winner_predictor import make_prediction

result = make_prediction(
    home_team="Boston Celtics",
    away_team="Los Angeles Lakers",
    date="2026-04-17"
)

print(f"Prediction: {result['prediction']}")
print(f"Home Win Probability: {result['home_probability']:.1%}")
print(f"Top Factors: {result['explanation']}")
```

## Model Details

### Architecture
- **Algorithm**: RandomForestClassifier (200 estimators)
- **Features**: 16 differential statistics (home vs. away team metrics)
- **Train/Test Split**: 80/20 chronological split (respects time order)
- **Class Weighting**: Balanced to handle class imbalance

### Performance Metrics
- **Test Accuracy**: ~57.7%
- **Precision (Home Win)**: 0.59
- **Recall (Home Win)**: 0.48
- **F1-Score (Home Win)**: 0.53

### Key Features for Prediction
1. Point differential (avg 5-game)
2. Offensive/defensive rating difference
3. Turnover rates
4. Three-point shooting efficiency
5. Player usage patterns
6. Fast break points

## Training the Model

To retrain on updated data:

```bash
python nba_winner_predictor.py
```

Or programmatically:

```python
from nba_winner_predictor import train_model
train_model()
```

### Training on a Subset (Fast Debug)

```bash
# Train on first 5000 games
$env:SAMPLE_N=5000
python nba_winner_predictor.py
```

## File Structure

```
ML/
├── app.py                           # Flask web server
├── nba_winner_predictor.py          # Core prediction pipeline
├── nba_winner_model.joblib          # Trained model artifact
├── templates/
│   └── index.html                   # Web UI
├── Data/
│   ├── Games.csv
│   ├── TeamStatistics.csv
│   ├── TeamStatisticsAdvanced.csv
│   ├── PlayerStatistics.csv
│   └── PlayerStatisticsUsage.csv
└── .venv/                           # Virtual environment
```

## Implementation Notes

### Data Pipeline
1. **Load**: Read games, team stats, advanced metrics, player stats
2. **Feature Engineering**: 
   - Compute lag-1 and rolling averages per team
   - Aggregate player stats (top-3 points, max usage)
   - Merge home/away features into matchup records
3. **Build**: Create differential features (home - away)
4. **Train**: RandomForest classifier on time-ordered split
5. **Predict**: Load model, compute features for upcoming game, return probability

### Handling Edge Cases
- Missing player stats filled with 0 (no contribution)
- Games with insufficient historical data flagged
- Time-series split prevents data leakage (training on future games)
- Balanced class weights to handle home team advantage

## Limitations & Improvements

### Current Limitations
- Home team advantage bias (~57% baseline accuracy from home team win rate)
- Limited to teams in historical data
- Date must be after team's first game in dataset
- Requires 5+ recent games for reliable features

### Potential Enhancements
- Add player injury/rest status
- Include seasonal form (momentum indicators)
- Ensemble multiple models (gradient boosting, neural nets)
- Add betting odds as input feature
- Cross-validation on multiple time-series folds
- Feature importance tracking over time

## Betting Strategy

⚠️ **Disclaimer**: Predictions are ~58% accurate—only ~8% above random chance.

### Responsible Use
- Use as one signal among many (combine with expert analysis, line movement, injury reports)
- Never bet more than you can afford to lose
- Set strict loss limits and stick to a bankroll strategy
- Track predictions vs. actuals to validate edge

## License

Project created for educational purposes.

## Contact

For questions or issues, refer to the data source documentation or adjust feature engineering in `build_matchup_dataset()`.
