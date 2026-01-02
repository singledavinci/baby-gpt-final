#!/bin/bash

# Create a swap file if it doesn't exist
if [ ! -f /tmp/swapfile ]; then
    fallocate -l 512M /tmp/swapfile
    chmod 600 /tmp/swapfile
    mkswap /tmp/swapfile
    swapon /tmp/swapfile
    echo "Swap created"
fi

# Run the application
exec gunicorn --workers 1 --threads 1 --timeout 120 app:app
