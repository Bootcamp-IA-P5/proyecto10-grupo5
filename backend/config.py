import os
from pathlib import Path

LOG_LEVEL = "DEBUG"
LOG_FILE_NAME = "backend.log"
LOG_BASE_DIR = "logs"

# Get the project root directory (parent of backend/)
PROJECT_ROOT = Path(__file__).parent.parent
PREFIX = str(PROJECT_ROOT / "resources" / "models") + "/"
