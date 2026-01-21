from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import sys

convert_zero_checkpoint_to_fp32_state_dict(sys.argv[1], sys.argv[2])

