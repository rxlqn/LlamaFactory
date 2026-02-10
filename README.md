## llamafactory 镜像启动

```bash
docker run -it --gpus=all --ipc=host -v /cpfs2:/cpfs2 /cpfs:/cpfs --name llamafactory docker.xuanyuan.run/hiyouga/llamafactory:latest
```

## 配置自定义环境

```bash
pip uninstall -y llamafactory
pip install -e ".[dev]"
```

## SFT 实验配置

本目录包含 Qwen3-VL-2B-Instruct 模型的 OCR SFT (Supervised Fine-Tuning) 实验配置。

### 模型配置

- **基础模型**: `/cpfs2/shared/models/Qwen3-VL-2B-Instruct`
- **图像最大分辨率**: 768x768 (589824 pixels)
- **视频最大分辨率**: 16384 pixels
- **模板**: `qwen3_vl_nothink`

### 训练方法

- **微调类型**: 全参数微调 (full finetuning)
- **DeepSpeed**: `examples/deepspeed/ds_z2_config.json`
- **混合精度**: bf16
- **学习率调度**: cosine
- **流式数据加载**: enabled

### 启动命令示例

```bash
# 单GPU调试
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=4,5,6,7 llamafactory-cli train workspace/sft/qwen3vl_full_sft_debug.yaml

# 训练脚本
bash workspace/sft/train.sh
```

### 输出目录

训练输出保存至 `saves/ocr_sft/` 目录，实验结果使用 W&B 进行记录。
