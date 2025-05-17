set -xu

runs="nwcsaf-rollout-1 trite-cabin cloudcast-production" 

python3 verify.py \
	--run_name $runs \
	--score mae mae2d psd

