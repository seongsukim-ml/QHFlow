import sys
import logging
import logging.handlers
import os
import json

try:
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: rich module not available, using basic logging")

# 설정 파일 경로
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "logger_config.json")

def load_config():
    """Load logger configuration from config file."""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file {CONFIG_FILE} not found, using default settings")
        return {
            "enable_logger": True,
            "use_rich": True,
            "log_level": "INFO",
            "log_file_path": "./log.log",
            "console_format": "%(levelname)s: %(message)s",
            "file_format": "[%(asctime)s]\\t%(levelname)s\\t[%(filename)s:%(funcName)s:%(lineno)s]\\t>> %(message)s",
            "rich_format": "[%(filename)s:%(lineno)s] >> %(message)s"
        }
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file {CONFIG_FILE}: {e}")
        return {
            "enable_logger": True,
            "use_rich": True,
            "log_level": "INFO",
            "log_file_path": "./log.log",
            "console_format": "%(levelname)s: %(message)s",
            "file_format": "[%(asctime)s]\\t%(levelname)s\\t[%(filename)s:%(funcName)s:%(lineno)s]\\t>> %(message)s",
            "rich_format": "[%(filename)s:%(lineno)s] >> %(message)s"
        }

# Load configuration
CONFIG = load_config()
ENABLE_LOGGER = CONFIG["enable_logger"]
USE_RICH = CONFIG["use_rich"] and RICH_AVAILABLE
LOG_PATH = CONFIG["log_file_path"]
RICH_FORMAT = CONFIG["rich_format"]
FILE_HANDLER_FORMAT = CONFIG["file_format"]
CONSOLE_FORMAT = CONFIG["console_format"]


# Global logger instance
_global_logger = None

def setup_global_logger():
    """Setup global logger once. This should be called at the start of the application."""
    global _global_logger
    
    if _global_logger is not None:
        return _global_logger
    
    if not ENABLE_LOGGER:
        # When logger is disabled, return basic console logger (no file logging)
        _global_logger = logging.getLogger("default")
        _global_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in _global_logger.handlers[:]:
            _global_logger.removeHandler(handler)
        
        # Add only basic console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))
        _global_logger.addHandler(console_handler)
        
        return _global_logger
    
    if USE_RICH:
        logging.basicConfig(
            level="NOTSET",
            format=RICH_FORMAT,
            handlers=[RichHandler(rich_tracebacks=True)]
        )
        _global_logger = logging.getLogger("rich")
    else:
        # Use basic console handler when rich is disabled
        logging.basicConfig(
            level="NOTSET",
            format=RICH_FORMAT,
            handlers=[logging.StreamHandler()]
        )
        _global_logger = logging.getLogger("basic")

    file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(FILE_HANDLER_FORMAT))
    _global_logger.addHandler(file_handler)

    return _global_logger

def get_logger(name: str = None) -> logging.Logger:
    """
    Get logger instance. If global logger is not setup, it will be setup automatically.
    
    Args:
        name: Optional logger name. If provided, returns a child logger of the global logger.
    
    Returns:
        Logger instance
    """
    global _global_logger
    
    if _global_logger is None:
        setup_global_logger()
    
    if name:
        return _global_logger.getChild(name)
    return _global_logger

def set_logger() -> logging.Logger:
    """Legacy function for backward compatibility. Use get_logger() instead."""
    return get_logger()

def handle_exception(exc_type, exc_value, exc_traceback):
    logger = get_logger()
    logger.error("Unexpected exception", exc_info=(exc_type, exc_value, exc_traceback))

if __name__ == "__main__":
    # Setup global logger
    setup_global_logger()
    sys.excepthook = handle_exception

    print(f"Config file: {CONFIG_FILE}")
    print(f"Logger enabled: {ENABLE_LOGGER}")
    print(f"Rich available: {RICH_AVAILABLE}")
    print(f"Use rich: {USE_RICH}")
    print(f"Log file path: {LOG_PATH}")
    
    # Test global logger
    logger = get_logger()
    logger.info("Testing global logger")
    
    # Test child logger
    child_logger = get_logger("test_module")
    child_logger.info("Testing child logger")
    
    for i in range(3, -1, -1):
        num = 1/i
        logger.info(f"1/{i} = {num}")