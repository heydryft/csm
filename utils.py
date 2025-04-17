import time
from datetime import datetime

# Add debug utility function with timing
def debug(message, start_time=None):
    """Debug function with optional timing information
    
    Args:
        message: The debug message to print
        start_time: Optional start time for timing calculations. If provided, elapsed time will be shown.
    """
    current_time = time.time()
    timestamp = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    if start_time is not None:
        elapsed = current_time - start_time
        formatted_message = f"[{timestamp}] {message} (elapsed: {elapsed:.4f}s)"
    else:
        formatted_message = f"[{timestamp}] {message}"
    
    print(formatted_message)
    return current_time  # Return current time so it can be used as start_time for subsequent calls