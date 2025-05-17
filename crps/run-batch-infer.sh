set -xu
export MODEL_FAMILY=pgu
export RUN_NAME=other-coda
export RUN_NUMBER=1
export ROLLOUT_LENGTH=5

python3 cc2trainer.py test \
	--config /data/runs/$RUN_NAME/$RUN_NUMBER/config.yaml \
	--ckpt_path /data/runs/$RUN_NAME/$RUN_NUMBER/checkpoints/last.ckpt \
	--model.init_weights_from_ckpt=False \
	--model.adapt_ckpt_resolution=False \
	--data.data_path='["../data/nwcsaf-475x535-202503-anemoi.zarr"]' \
	--trainer.logger=False \
	--trainer.devices=1 \
	--trainer.strategy=auto \
	--data.test_split=1 \
	--data.val_split=0 \
	--data.batch_size=8 \
	--model.rollout_length=$ROLLOUT_LENGTH \
	--data.rollout_length=$ROLLOUT_LENGTH

