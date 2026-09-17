from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent / 'ML'
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
