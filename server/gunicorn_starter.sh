#!/bin/sh

gunicorn --bind '0.0.0.0:8080' --workers 1 --threads 24 --timeout=0 server.app:app