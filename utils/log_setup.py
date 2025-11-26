import logging
import logging.config
from pathlib import Path

from backend.config import LOG_FILE_NAME, LOG_LEVEL, LOG_BASE_DIR

def log_setup():
    # Ensure the log directory exists
    log_path = Path(LOG_FILE_NAME)
    if not log_path.is_absolute():
        log_path = Path(LOG_BASE_DIR) / log_path
        
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    color_scheme = {
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    }

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] - %(filename)s - %(funcName)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'colored_console': {
                '()': 'colorlog.ColoredFormatter', # Use the colorlog class
                'format': '%(log_color)s%(asctime)s [%(levelname)s] - %(filename)s - %(funcName)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
                'log_colors': color_scheme,
            },
        },
        'handlers': {
            'file': {
                'level': LOG_LEVEL,
                # Use RotatingFileHandler to prevent disk fill-up
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': str(log_path),
                'maxBytes': 1024 * 1024 * 5, # 5 MB
                'backupCount': 5,            # Keep 5 backup files
                'formatter': 'standard',
                'encoding': 'utf-8',
            },
            'console': {
                'level': LOG_LEVEL,
                'class': 'logging.StreamHandler',
                'formatter': 'colored_console',
            },
        },
        'loggers': {
            # The root logger catches everything
            '': { 
                'handlers': ['file', 'console'],
                'level': LOG_LEVEL,
                'propagate': True,
            },
            'watchfiles': {
                'handlers': ['console'], # Optional: Keep it in console if you want, remove 'file'
                'level': 'WARNING',      # Only show warnings/errors, hide DEBUG changes
                'propagate': False,      # Stop it from bubbling up to the root logger
            },
            # You might also want to silence uvicorn access logs if they are too noisy
            'uvicorn.access': {
                'handlers': ['console'],
                'level': 'WARNING',
                'propagate': False,
            },
            # urllib3
            'urllib3': {
                'handlers': ['console'],
                'level': 'WARNING',
                'propagate': False,
            }
        }
    }

    logging.config.dictConfig(logging_config)