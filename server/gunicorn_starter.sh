#!/bin/sh

# export PYTHONPATH=src

gunicorn --bind '0.0.0.0:8080' --worker-class 'gevent' --workers 4 --threads 4 --timeout=0 server.app:app