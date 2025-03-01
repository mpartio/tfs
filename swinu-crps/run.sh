export PYTHONUNBUFFERED=1

if [ -s log/train.log ]; then
  mv log/train.log log/train.log.$(date +%Y%m%d%H%M%S)
fi

set -uex

# rm -f figures/*.png
# export CUDA_VISIBLE_DEVICES=1

nohup python3 cc2CRPS-L.py \
	--data_path ../data/nwcsaf-238x268-hourly-anemoi.zarr \
	--num_data_channels 1 \
	--num_forcing_channels 9 \
	--rollout_length 1 \
	--num_devices 2 \
	--batch_size 8 \
	--patch_size 8 \
	--hidden_dim 192 \
	--precision 16-mixed \
	--input_resolution 268 238 \
	--num_iterations 100000 \
	--learning_rate 1e-4 \
	--strategy ddp_find_unused_parameters_true \
	--generate_run_name > log/train.log 2>&1 &

sleep 1
tail -f log/train.log

