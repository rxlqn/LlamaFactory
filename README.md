## llamafactory 镜像启动

```bash
docker run -it --gpus=all --ipc=host -v /cpfs2:/cpfs2 /cpfs:/cpfs --name llamafactory docker.xuanyuan.run/hiyouga/llamafactory:latest
```

## 配置自定义环境

### 通用配置

```bash
pip uninstall -y llamafactory
pip install -e ".[dev]"
```

### Qwen3.5 环境配置

由于 Qwen3.5 新增了 linear attention，需要以下特殊依赖：

```bash
# 1. 安装 PyTorch 2.7.1 (CUDA 12.8)
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. 安装 Triton 3.3.1 (与 PyTorch 2.7.1 兼容，通常默认在step1与torch一同安装了)
# pip install triton==3.3.1

# 3. 安装 Transformers 5.2.0
pip install transformers==5.2.0

# 4. 安装 PEFT (>= 0.18.1)
pip install peft>=0.18.1

# 5. 安装 flash-linear-attention > 0.4.0
git clone https://github.com/fla-org/flash-linear-attention
cd flash-linear-attention
pip install -U .

# 6. 安装 causal-conv1d
pip install causal-conv1d --no-binary :all: --no-build-isolation
```

## SFT 实验配置

本目录包含 OCR SFT (Supervised Fine-Tuning) 实验配置，支持以下模型：

### 1. Qwen3-VL-2B-Instruct

#### 模型配置

- **基础模型**: `/cpfs2/shared/models/Qwen3-VL-2B-Instruct`
- **图像最大分辨率**: 768x768 (589824 pixels)
- **视频最大分辨率**: 16384 pixels
- **模板**: `qwen3_vl_nothink`

#### 训练配置

- **微调类型**: 全参数微调 (full finetuning)
- **DeepSpeed**: `examples/deepspeed/ds_z2_config.json`
- **混合精度**: bf16
- **学习率调度**: cosine
- **流式数据加载**: enabled
- **数据集**: `ocrsft_3_5_html_train`

#### 启动命令

```bash
# 单GPU调试
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0 llamafactory-cli train workspace/sft/qwen3vl_full_sft_debug.yaml

# 多GPU训练
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train workspace/sft/0213/exp1.yaml
```

---

### 2. Qwen3.5-2B

#### 模型配置

- **基础模型**: `/cpfs2/shared/models/Qwen3.5/Qwen3.5-2B`
- **图像最大分辨率**: 768x768 (589824 pixels)
- **视频最大分辨率**: 16384 pixels
- **模板**: `qwen3_5_nothink`

#### 训练配置

- **微调类型**: 全参数微调 (full finetuning)
- **DeepSpeed**: `examples/deepspeed/ds_z2_config.json`
- **混合精度**: bf16
- **Flash Attention**: fa2
- **学习率调度**: cosine
- **学习率**: 3.0e-6
- **流式数据加载**: enabled
- **序列长度**: 8192
- **数据集**: `ocrsft_4_0_html_train`

#### 训练参数 (4 GPUs)

- **per_device_train_batch_size**: 2
- **gradient_accumulation_steps**: 8
- **max_steps**: 40633
- **warmup_ratio**: 0.05

#### 启动命令

```bash
# 单GPU调试
DISABLE_VERSION_CHECK=1 FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0 llamafactory-cli train workspace/sft/qwen3_5_full_sft_debug.yaml

# 多GPU训练
DISABLE_VERSION_CHECK=1 FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train workspace/sft/0303/exp1.yaml
```

---

### 训练脚本

```bash
bash workspace/sft/train.sh
```

### 输出目录

训练输出保存至 `saves/ocr_sft/` 目录，实验结果使用 W&B 进行记录。
