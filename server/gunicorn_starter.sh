#!/bin/sh

# gunicorn server.app:app -w 2 --threads 6 -b '0.0.0.0:8080'
# export PYTHONPATH=src

gunicorn --bind '0.0.0.0:8080' --workers 1 --threads 6 server.app:app
# gunicorn --bind '127.0.0.1:8080' --workers 1 --chdir . server.app_noc:app
# worker times out for doc retrieval if not threaded.