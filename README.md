# HUM

Official implementation of the paper:
**Heterogeneous User Modeling for LLM-based Recommendation**
[RecSys'25 Oral](https://arxiv.org/abs/2507.04626)

## 📖 Overview

Leveraging Large Language Models (LLMs) for recommendation has demonstrated notable success in various domains. A key challenge lies in effectively modeling user preferences from heterogeneous behaviors across multiple domains. **HUM (Heterogeneous User Modeling)** addresses this by incorporating:

- **Compression Enhancer**: Uses a customized prompt to compress heterogeneous behaviors into a tailored token.
- **Robustness Enhancer**: Introduces a domain importance score to mitigate the "domain seesaw" phenomenon by guiding domain optimization.

Extensive experiments on heterogeneous datasets validate that HUM effectively models user heterogeneity, achieving both high efficacy and robustness.

## 🚀 Getting Started

### 1. Installation

Ensure you have the following dependencies installed:

```bash
pip install torch transformers peft omegaconf tqdm wandb
```

### 2. Data Preparation

Download the processed datasets from the following link:
[Google Drive Download](https://drive.google.com/drive/folders/1ryvaZwK9n_kDL1N2zQg5rX8g7Jhb_kby?usp=drive_link)

Place the downloaded data into the `dataset/` directory:
```text
HUM/
├── dataset/
│   └── [dataset_name]/
│       ├── single_domain_iid.pkl
│       ├── inverted_iid.pkl
│       └── ...
```

### 3. Training

To start training the HUM model, use the provided shell script:

```bash
bash scripts/train_HUM.sh [OUTPUT_PATH] [DATASET] [LR] [CONFIG] [NUM_GPUS] [PORT]
```

Example:
```bash
bash scripts/train_HUM.sh ./ckp/hum_v1.pth m_IOATBC 5e-5 configs/train_HUM.yaml 4 29500
```

### 4. Configuration

The training parameters can be adjusted in `configs/train_HUM.yaml`. Key parameters include:
- `backbone`: The base LLM (e.g., Qwen2.5-1.5B).
- `batch_size`: Training batch size.
- `lora_r`, `lora_alpha`: LoRA configuration.
- `num_gpus`: Number of GPUs for distributed training.

## 📂 Project Structure

- `hum.py`: Core model implementation (HUM class).
- `HUM_Trainer.py`: Training logic and loop.
- `dataloader.py`: Data loading and preprocessing.
- `hum_logger.py`: Unified logging system.
- `param.py` & `param_parse.py`: Argument parsing and configuration management.
- `utils.py` & `metrics.py`: Utility functions and evaluation metrics.
- `parse_result.py`: Script for parsing log files and extracting results.

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{bao2025heterogeneous,
  title={Heterogeneous user modeling for llm-based recommendation},
  author={Bao, Honghui and Wang, Wenjie and Lin, Xinyu and Zhu, Fengbin and Sun, Teng and Feng, Fuli and Chua, Tat-Seng},
  booktitle={RecSys},
  pages={145--154},
  year={2025}
}
```