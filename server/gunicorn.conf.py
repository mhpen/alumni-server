import os

# Get port from environment variable or default to 10000
port = int(os.environ.get("PORT", 10000))

# Bind to 0.0.0.0 to listen on all interfaces
bind = f"0.0.0.0:{port}"

# Number of worker processes
workers = 4

# Timeout in seconds
timeout = 120

# Log level
loglevel = "info"

# Access log format
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr

# Preload the application
preload_app = True

# Maximum number of requests a worker will process before restarting
max_requests = 1000
max_requests_jitter = 50

# Process name
proc_name = "alumni_server"
