# export CC2_RUN_NAME=$(sh generate-random-name.sh)

config=config.yaml

# 1. Start from scratch
# python3 cc2trainer.py fit --config $config

# 2. Continue training from a checkpoint including optimizer state
# python3 cc2trainer.py fit --config $config --ckpt_path ./runs/$CC2_RUN_NAME/checkpoints/last.ckpt

# 3. Continue training from a checkpoint excluding optimizer state
# python3 cc2trainer.py fit --config $config --model.init_weights_from_ckpt=true

# 4. Continue training from a checkpoint excluding optimizer state and adapt checkpoint to model
#python3 cc2trainer.py fit --config $config --model.init_weights_from_ckpt=true --model.adapt_ckpt_resolution=true
