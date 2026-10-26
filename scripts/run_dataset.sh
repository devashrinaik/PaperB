#!/usr/bin/env bash
# Extract everything for one dataset across both GPUs, in the agreed order:
# primary model (ViT-g) primary configs first, then variants and replications; LTX-Video last.
#   scripts/run_dataset.sh robolab   |   scripts/run_dataset.sh ssv2
set -eo pipefail
DS=$1
ROOT=/mnt/ssd1/devashri/CVPR
source /home/devashri/miniconda3/etc/profile.d/conda.sh; conda activate paperB
cd $ROOT; mkdir -p logs
X="python -W ignore scripts/extract.py --dataset $DS"
if [ "$DS" = "robolab" ] || [ "$DS" = "synthetic_robolab" ]; then
  PRIM=s3o0w4x1,s3o1w4x1,s3o2w4x1          # K1 offset 0 + K2 merge streams
  VAR=s3o0w8x2,s3o1w8x2,s3o2w8x2,s1o0w4x1
else
  PRIM=s3o0w4x1
  VAR=s3o0w8x2,s1o0w4x1
fi
# phase 1: primary model on both GPUs (sharded) + flow
( $X --model vjepa2_vitg --gpu 0 --configs $PRIM --shard 0/2 && \
  $X --model vjepa2_vitg --gpu 0 --configs $VAR --shard 0/2 ) > logs/${DS}_vitg_0.log 2>&1 &
( $X --model raft --gpu 1 && \
  $X --model vjepa2_vitg --gpu 1 --configs $PRIM --shard 1/2 && \
  $X --model vjepa2_vitg --gpu 1 --configs $VAR --shard 1/2 ) > logs/${DS}_raft_vitg_1.log 2>&1 &
wait
echo "phase 1 (ViT-g + flow) done: run K1/K2 on the primary model now"
# phase 2: replications
( $X --model vjepa2_vitl --gpu 0 --configs $PRIM,$VAR ) > logs/${DS}_vitl.log 2>&1 &
( $X --model vjepa2_1_vitg --gpu 1 --configs $PRIM --shard 0/1 ) > logs/${DS}_v21.log 2>&1 &
wait
# phase 3: LTX-Video, last
( $X --model ltx --gpu 0 --shard 0/2 ) > logs/${DS}_ltx_0.log 2>&1 &
( $X --model ltx --gpu 1 --shard 1/2 ) > logs/${DS}_ltx_1.log 2>&1 &
wait
echo "all extraction done for $DS"
