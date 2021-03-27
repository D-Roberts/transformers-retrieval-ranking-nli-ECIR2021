#!/bin/sh

# export PYTHONPATH=src

gunicorn --bind '0.0.0.0:8080' --worker-class 'gevent' --workers 2 --threads 1 --timeout=0 server.app:app