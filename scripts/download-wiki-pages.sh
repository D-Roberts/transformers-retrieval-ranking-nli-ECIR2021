#!/bin/bash
wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
unzip wiki-pages.zip -d data

# Then:
# PYTHONPATH=src python3 scripts/build_db.py data/wiki-pages data/fever/fever.db

# Cleanup 
# rm wiki-pages.zip
# rm -r __MACOSX/
# rm -r data/wiki-pages

# License Wikipedia included
