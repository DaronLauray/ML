from pathlib import Path

ROOT = Path(__file__).resolve().parent


def test_pages_model_assets_exist():
    assert (ROOT / 'model_export.json').exists(), 'GitHub Pages model export is missing'
    assert (ROOT / 'team_profiles.json').exists(), 'GitHub Pages team profile data is missing'
