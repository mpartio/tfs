export PYTHONUNBUFFERED=1

if [ -s log/train.log ]; then
  mv log/train.log log/train.log.$(date +%Y%m%d%H%M%S)
fi

set -uex

# rm -f figures/*.png
nohup python3 cc2CRPS-L.py --apply_smoothing \
	--num_layers 1 \
	--num_blocks 12 8 4 4 8 12 \
	--num_heads 8 4 4 4 4 8 \
	--precision 16-mixed \
	--hidden_dim 256 \
	--input_resolution 128 128 \
	--data_path ../data/nwcsaf-128x128.zarr \
	--num_members 3 \
	--num_iterations 300000 \
	--learning_rate 1e-3 \
	--rollout_length 1 \
	--batch_size 12 > log/train.log 2>&1 &
#	--only_config \
#	--run_name excited-fjord \

sleep 1
tail -f log/train.log

