#!/bin/bash

# Start the metric server in the background
#python3 -u metric_server.py &

# Start the websocket server in the foreground
python3 -u websocket_server.py --port 7000 --workers 5
