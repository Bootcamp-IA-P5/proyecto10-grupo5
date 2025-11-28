import os
from dotenv import load_dotenv

# Load environment variables from a .env file (optional, but recommended for development)
# You would need to install python-dotenv: pip install python-dotenv
try:
    load_dotenv()
except ImportError:
    pass

# Get the backend URL from an environment variable or use a default
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# API_URL is often the same as BACKEND_URL, but you can separate them if needed.
# For this app, we'll keep the logic simple as in the original code.
API_URL = os.getenv("API_URL", f"{BACKEND_URL}/docs")

# --- App Settings ---
APP_TITLE = "Toxicity Classifier"
PAGE_ICON = "🤖"

# --- Logging Settings ---
LOG_LEVEL="DEBUG"
LOG_FILE_NAME="frontend.log"
LOG_BASE_DIR="logs"