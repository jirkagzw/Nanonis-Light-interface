import logging
import inspect
import os
import threading
from datetime import datetime
from functools import wraps

_logger_initialized = False
_call_depth = threading.local()  # thread-local depth counter

def init_logger(session_path: str):
    global _logger_initialized
    if _logger_initialized:
        print("Logger already initialized")
        return

    os.makedirs(session_path, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(session_path, f"log_{date_str}.log")
    print(f"Writing logs to: {log_file}")

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        print("Clearing existing handlers")
        root_logger.handlers.clear()

    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    _logger_initialized = True


def log_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Build argument representation
            # Remove 'self' from args if method
            args_without_self = args[1:] if args else []
            arg_list = [repr(a) for a in args_without_self]
            kwarg_list = [f"{k}={v!r}" for k, v in kwargs.items()]
            all_args = ", ".join(arg_list + kwarg_list)
            logging.info(f"Called: {func.__qualname__}({all_args})")
        except Exception as e:
            logging.error(f"Logging failed in {func.__qualname__}: {e}")
        return func(*args, **kwargs)
    return wrapper


def log_calls(max_depth=0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(_call_depth, 'level'):
                _call_depth.level = 0

            # Check if this is a method (self is first arg) and if logging is enabled
            instance = args[0] if args else None
            logging_enabled = getattr(instance, "logging_enabled", True)

            if logging_enabled and _call_depth.level < max_depth:
                try:
                    args_without_self = args[1:] if args else []
                    arg_list = [repr(a) for a in args_without_self]
                    kwarg_list = [f"{k}={v!r}" for k, v in kwargs.items()]
                    all_args = ", ".join(arg_list + kwarg_list)
                    logging.info(f"Called: {func.__qualname__}({all_args})")
                except Exception as e:
                    logging.error(f"Logging failed in {func.__qualname__}: {e}")

            _call_depth.level += 1
            try:
                return func(*args, **kwargs)
            finally:
                _call_depth.level -= 1
        return wrapper
    return decorator

def apply_logging(cls, max_depth=1):
    """
    Apply logging decorator to all public callable methods of a class.
    Uses log_calls with max_depth for nested call filtering.
    """
    for attr_name in dir(cls):
        if attr_name.startswith("_"):
            continue  # skip private/protected/dunder methods
        attr = getattr(cls, attr_name)
        if callable(attr):
            # Wrap with log_calls decorator with max_depth
            decorated = log_calls(max_depth=max_depth)(attr)
            setattr(cls, attr_name, decorated)
    return cls