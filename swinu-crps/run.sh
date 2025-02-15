export PYTHONUNBUFFERED=1

if [ -s log/train.log ]; then
  mv log/train.log log/train.log.$(date +%Y%m%d%H%M%S)
fi

set -uex

# rm -f figures/*.png
nohup python3 cc2CRPS-L.py --apply_smoothing \
	--num_layers 0 \
	--num_blocks 12 8 4 4 8 12 \
	--num_heads 8 4 4 4 4 8 \
	--precision 16-mixed \
	--hidden_dim 128 \
	--input_resolution 64 64 \
	--data_path ../data/nwcsaf-64x64.zarr \
	--num_members 3 \
	--num_iterations 300000 \
	--learning_rate 1e-4 \
	--run_name full-offset \
	--rollout_length 3 \
	--batch_size 12 > log/train.log 2>&1 &

sleep 1
tail -f log/train.log

