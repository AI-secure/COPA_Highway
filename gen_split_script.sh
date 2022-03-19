set -x
python gen_split.py --train-data-folder /data1/common/chejian/copa/dqn_replay/$1/$2/replay_logs \
	                    --epi-index-path /data1/common/chejian/copa/dqn_replay/hash_split/$1_$2/partition_$3.pt \
			                        --output-folder /data1/common/chejian/copa/dqn_replay/hash_split/$1_$2/dataset/hash_$3 \
						                    --start-id 0 --end-id 400

