#!/bin/sh

export PYTHONPATH=src
gunicorn --bind '0.0.0.0:8080' --workers 1 --threads 36 --chdir . --timeout=0 server.app:app