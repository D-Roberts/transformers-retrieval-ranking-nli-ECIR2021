# build wiki pages db to match annotations fever

wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
unzip wiki-pages.zip -d data

# Then:
# PYTHONPATH=src python build_datasets_scripts/build_db.py data/wiki-pages data/fever/fever.db

# Cleanup 
# rm wiki-pages.zip
# rm -r __MACOSX/
# rm -r data/wiki-pages

# License Wikipedia included
