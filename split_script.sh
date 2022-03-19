set -x
python split.py --train-data-folder /data1/common/chejian/copa/dqn_replay/$1/$2/replay_logs \
	                    --output-folder /data1/common/chejian/copa/dqn_replay/hash_split/$1_$2


