#!/bin/bash
# lora
# deepspeed
set -x

export TOKENIZERS_PARALLELISM=false

# FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=2,3 llamafactory-cli train workspace/sft/0202/exp1.yaml
# FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=5,6 llamafactory-cli train workspace/sft/0130/exp4.yaml
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=4,5,6,7 llamafactory-cli train workspace/sft/0207/exp2.yaml