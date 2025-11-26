"""
Centralized Logging Configuration for StableVITON
Creates detailed logs for tracking all operations, errors, and debugging
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.DEBUG):
    """
    Set up a logger with both file and console handlers
    
    Args:
        name: Logger name (usually __name__ of the module)
        log_file: Optional specific log file path
        level: Logging level (default: DEBUG)
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG and above) - all details
    if log_file is None:
        log_dir = Path(__file__).parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f'stableviton_{datetime.now().strftime("%Y%m%d")}.log'
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    return logger


def setup_app_logger():
    """Set up logger specifically for Flask app"""
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'flask_app_{datetime.now().strftime("%Y%m%d")}.log'
    return setup_logger('flask_app', log_file)


def setup_inference_logger():
    """Set up logger specifically for inference"""
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'inference_{datetime.now().strftime("%Y%m%d")}.log'
    return setup_logger('inference', log_file)


def setup_preprocessing_logger():
    """Set up logger specifically for preprocessing"""
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'preprocessing_{datetime.now().strftime("%Y%m%d")}.log'
    return setup_logger('preprocessing', log_file)


def log_exception(logger, exc, context=""):
    """
    Log exception with full context and traceback
    
    Args:
        logger: Logger instance
        exc: Exception object
        context: Additional context string
    """
    import traceback
    
    error_msg = f"{context}\n" if context else ""
    error_msg += f"Exception Type: {type(exc).__name__}\n"
    error_msg += f"Exception Message: {str(exc)}\n"
    error_msg += "Traceback:\n"
    error_msg += "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    
    logger.error(error_msg)


def log_gpu_status(logger):
    """Log current GPU memory status"""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                logger.info(f"GPU {i} Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
        else:
            logger.info("No GPU available")
    except ImportError:
        logger.debug("PyTorch not available, skipping GPU status")
    except Exception as e:
        logger.warning(f"Could not get GPU status: {e}")
