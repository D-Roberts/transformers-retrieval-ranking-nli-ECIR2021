mkdir data
mkdir data/fever-data

# From https://github.com/UKPLab/fever-2018-team-athene
wget -O data/fever-data/train.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
wget -O data/fever-data/dev.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl
wget -O data/fever-data/test.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl

# license of fever.ai