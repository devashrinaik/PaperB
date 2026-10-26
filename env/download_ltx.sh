#!/usr/bin/env bash
set -eo pipefail
source /home/devashri/miniconda3/etc/profile.d/conda.sh; conda activate paperB
huggingface-cli download Lightricks/LTX-Video \
  --include "model_index.json" --include "transformer/*" --include "vae/*" \
  --include "scheduler/*" --include "text_encoder/*" --include "tokenizer/*" \
  --local-dir /mnt/ssd1/devashri/CVPR/weights/LTX-Video
echo LTX_DONE
