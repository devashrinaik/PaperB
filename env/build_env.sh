#!/usr/bin/env bash
# Paper B env: conda env "paperB" (python 3.12, torch 2.5.1 cu121; driver 550 / CUDA 12.4)
set -eo pipefail
source /home/devashri/miniconda3/etc/profile.d/conda.sh
ROOT=/mnt/ssd1/devashri/CVPR
conda create -y -n paperB python=3.12
conda activate paperB
python -m pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
# fresh vjepa2 checkout pinned to the commit already on this server (includes V-JEPA 2.1)
[ -d $ROOT/third_party/vjepa2 ] || git clone https://github.com/facebookresearch/vjepa2.git $ROOT/third_party/vjepa2
git -C $ROOT/third_party/vjepa2 checkout 204698b
pip install -r $ROOT/third_party/vjepa2/requirements.txt
pip install -e $ROOT/third_party/vjepa2
pip install scipy scikit-learn pandas pyarrow matplotlib av
python -c "import torch,torchvision; print('torch', torch.__version__, torchvision.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import src.models.vision_transformer, src.models.predictor; print('vjepa2 import ok')" || python -c "import app; print('vjepa2 app import ok')"
echo BUILD_ENV_DONE
# vjepa2 editable install does not expose top-level `src`; add the repo to sys.path
echo $ROOT/third_party/vjepa2 > $(python -c "import site;print(site.getsitepackages()[0])")/vjepa2_repo.pth
# LTX-Video (diffusers) for the YoCausal-style denoising-loss baseline
pip install diffusers==0.42.0 transformers accelerate sentencepiece safetensors huggingface_hub
bash $ROOT/env/download_ltx.sh
